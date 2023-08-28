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

from dalle2_video.video_encoder import ViViT
from dalle2_video.datasets import CelebVTextDataset, CelebVTextCollator
from dalle2_video.utils import CLIPLoss, Classifier, sequential_apply


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

    run_dir = os.path.join("runs/celebv-text", args.train_name)
    os.makedirs(run_dir, exist_ok=True)

    device = f"cuda:{args.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = CelebVTextDataset(
        texts_path=os.path.join(args.texts_dirs.root, "tokenized.pt"),
        videos_path=os.path.join(args.videos_dirs.root, "preprocessed.h5"),
    )

    train_size = int(len(dataset) * args.train_ratio)
    test_size = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset,
        lengths=[train_size, test_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate_fn = CelebVTextCollator(dataset.videos_ref)

    loader_args = {
        "collate_fn": collate_fn,
        "drop_last": True,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=args.batch_size, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=test_size, shuffle=False, **loader_args
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
    min_test_loss = float("inf")

    for epoch in range(args.epochs):
        train_losses = []
        test_losses = []
        train_top10_accs = []
        train_top1_accs = []
        test_top10_accs = []
        test_top1_accs = []
        inference_times = []

        video_encoder.train()
        if args.accum_grad:
            optimizer.zero_grad()

        for texts, videos in tqdm(train_loader, desc="Train"):
            texts, videos = texts.to(device), videos.to(device)

            text_embeds = clip_model.encode_text(texts)

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
                stime = time()

                # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].

                text_embeds = sequential_apply(
                    texts,
                    clip_model.encode_text,
                    args.batch_size,
                    desc="CLIP text encoder",
                )

                video_embeds = sequential_apply(
                    videos,
                    video_encoder,
                    args.batch_size,
                    desc="Video encoder",
                )

                inference_times.append(time() - stime)

                loss = loss_func(video_embeds, text_embeds)

                test_top1_acc, test_top10_acc, _ = classifier(
                    text_embeds, video_embeds, sequential=args.test_with_whole
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
                "VisionEncoder avg inference time": np.mean(inference_times),
            }
            wandb.log(performance_now)

        scheduler.step()

        trained_models.save(run_dir)

        if np.mean(test_losses) < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            trained_models.save(run_dir, best=True)

            min_test_loss = np.mean(test_losses)


if __name__ == "__main__":
    run()
