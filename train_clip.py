import os, sys
import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from termcolor import cprint
from time import time

import clip
from accelerate import Accelerator, DistributedType

from dalle2_video.video_encoder import ViViT
from dalle2_video.datasets import CelebVTextDataset, CelebVTextCollator
from dalle2_video.utils import CLIPLoss, Classifier, sequential_apply

CAST_TYPE_MAP = {"fp16": torch.half, "bp16": torch.bfloat16, "no": torch.float}


@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def run(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.init(
            project="dalle2-video",
            config=OmegaConf.to_container(args),
            save_code=True,
        )
        wandb.run.name = args.train_name
        wandb.run.save()

    run_dir = os.path.join("runs/celebv-text", args.train_name, "clip")
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = CelebVTextDataset(
        texts_path=os.path.join(args.texts_dirs.tokenized),
        videos_path=os.path.join(args.videos_dirs.preprocessed),
    )

    train_size = int(len(dataset) * args.train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
        # NOTE: it is crucial to set generator seed to keep train/test split consistent across
        #       CLIP/prior/decoder training.
    )

    loader_args = {
        "batch_size": args.batch_size,
        "collate_fn": dataset.collate_fn,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, shuffle=False, **loader_args
    )

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)

    # ---------------------
    #        Models
    # ---------------------
    video_encoder = ViViT(num_frames=args.seq_len * args.fps, **args.video_encoder).to(device)  # fmt: skip

    clip_model, _ = clip.load(args.clip_model, device=device)

    classifier = Classifier(args)

    # ---------------------
    #      Optimizers
    # ---------------------
    optimizer = torch.optim.Adam(video_encoder.parameters(), lr=args.lr)

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )
    elif args.lr_scheduler == "multistep":
        mlstns = [int(m * args.epochs) for m in args.lr_multistep_mlstns]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=mlstns, gamma=args.lr_step_gamma
        )
    else:
        raise ValueError()

    # -----------------------
    #     Strat training
    # -----------------------
    accelerator = Accelerator()

    accelerator.gradient_accumulation_steps = args.deepspeed.gradient_accumulation_steps

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        precision = CAST_TYPE_MAP[accelerator.mixed_precision]
        assert (precision == torch.float), "DeepSpeed currently only supports float32 precision when using on the fly embedding generation from clip"  # fmt: skip
        clip_model.to(precision)

    video_encoder, optimizer, scheduler, train_loader, test_loader = accelerator.prepare(
        video_encoder, optimizer, scheduler, train_loader, test_loader
    )

    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []
        train_top10_accs = []
        train_top1_accs = []
        test_top10_accs = []
        test_top1_accs = []

        video_encoder.train()
        if args.accum_grad:
            optimizer.zero_grad()

        for texts, videos in tqdm(train_loader, desc="Train"):
            texts, videos = texts.to(device), videos.to(device)

            with torch.no_grad():
                # NOTE: CLIP model output is originally fp16
                text_embeds = clip_model.encode_text(texts).float()

            video_embeds = video_encoder(videos)

            loss = loss_func(video_embeds, text_embeds)

            with torch.no_grad():
                train_top1_acc, train_top10_acc, _ = classifier(text_embeds, video_embeds)

            train_losses.append(loss.item())
            train_top10_accs.append(train_top10_acc)
            train_top1_accs.append(train_top1_acc)

            if args.accum_grad:
                loss.backward()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if args.accum_grad:
            optimizer.step()

        video_encoder.eval()
        for texts, videos in tqdm(test_loader, desc="Test"):
            texts, videos = texts.to(device), videos.to(device)

            with torch.no_grad():
                text_embeds = clip_model.encode_text(texts).float()

                video_embeds = video_encoder(videos)

                loss = loss_func(video_embeds, text_embeds)

                test_top1_acc, test_top10_acc, _ = classifier(
                    text_embeds, video_embeds, sequential=False
                )

            test_losses.append(loss.item())
            test_top10_accs.append(test_top10_acc)
            test_top1_accs.append(test_top1_acc)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss: {np.mean(train_losses):.3f} | ",
            f"avg test loss: {np.mean(test_losses):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "train_top10_acc": np.mean(train_top10_accs),
                "train_top1_acc": np.mean(train_top1_accs),
                "test_top10_acc": np.mean(test_top10_accs),
                "test_top1_acc": np.mean(test_top1_accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item(),
            }
            wandb.log(performance_now)

        scheduler.step()

        torch.save(video_encoder.state_dict(), os.path.join(run_dir, "video_encoder_last.pt"))  # fmt: skip

        if np.mean(test_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(video_encoder.state_dict(), os.path.join(run_dir, "video_encoder_best.pt"))  # fmt: skip

            min_test_loss = np.mean(test_losses)


if __name__ == "__main__":
    run()
