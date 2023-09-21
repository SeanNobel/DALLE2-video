import os, sys
import numpy as np
import torch
import torch.nn as nn
from torchinfo import summary
from time import time
from tqdm import tqdm
from termcolor import cprint
from typing import Union, Optional
import logging
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from accelerate import Accelerator, DeepSpeedPlugin

from dalle2_video.datasets import CelebVTextDataset
from dalle2_video.dalle2_video import Unet3D, VideoDecoder
from dalle2_video.trainer import VideoDecoderTrainer


@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def run(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.use_wandb:
        wandb.config = {
            k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]
        }
        wandb.init(
            project="dalle2-video_decoder",
            config=OmegaConf.to_container(args),
            save_code=True,
        )
        wandb.run.name = args.train_name
        wandb.run.save()

    run_dir = os.path.join("runs/celebv-text", args.train_name, "decoder")
    os.makedirs(run_dir, exist_ok=True)

    accelerator = Accelerator()
    accelerator.gradient_accumulation_steps = args.deepspeed.gradient_accumulation_steps

    device = accelerator.device

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = CelebVTextDataset(
        videos_path=args.videos_dirs.preprocessed,
        video_embeds_path=args.videos_dirs.embed,
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
        "batch_size": args.decoder.batch_size,
        "collate_fn": dataset.collate_fn,
        "drop_last": True,
        "num_workers": args.decoder.num_workers,
        "pin_memory": True,
    }
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, shuffle=True, **loader_args
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, shuffle=False, **loader_args
    )

    # ---------------------
    #        Models
    # ---------------------
    unet1 = Unet3D(
        dim=args.unet1.dim,
        video_embed_dim=args.dim,
        channels=args.channels,
        dim_mults=tuple(args.unet1.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    unet2 = Unet3D(
        dim=args.unet2.dim,
        video_embed_dim=args.dim,
        channels=args.channels,
        dim_mults=tuple(args.unet2.dim_mults),
        cond_on_text_encodings=False,
    ).to(device)

    decoder = VideoDecoder(
        unet=(unet1, unet2),
        frame_sizes=tuple(args.frame_sizes),
        frame_numbers=tuple(args.frame_numbers),
        timesteps=args.timesteps,
        learned_variance=False,
    ).to(device)

    # ---------------------
    #        Trainer
    # ---------------------
    decoder_trainer = VideoDecoderTrainer(
        decoder,
        accelerator=accelerator,
        dataloaders={
            "train": train_loader,
            "val": test_loader,
        },
        **args.decoder_trainer,
    )

    # -----------------------
    #     Strat training
    # -----------------------
    min_test_loss = float("inf")

    for epoch in range(args.decoder.epochs):
        train_losses_unet1 = []
        train_losses_unet2 = []
        test_losses_unet1 = []
        test_losses_unet2 = []

        for video_embed, video in tqdm(decoder_trainer.train_loader):
            video_embed, video = video_embed.to(device), video.to(device)

            loss_unet1 = decoder_trainer(
                video_embed=video_embed, video=video, unet_number=1
            )
            decoder_trainer.update(1)

            loss_unet2 = decoder_trainer(
                video_embed=video_embed, video=video, unet_number=2
            )
            decoder_trainer.update(2)

            train_losses_unet1.append(loss_unet1)
            train_losses_unet2.append(loss_unet2)

        for video_embed, video in tqdm(decoder_trainer.val_loader):
            video_embed, video = video_embed.to(device), video.to(device)

            loss_unet1 = decoder_trainer(
                video_embed=video_embed, video=video, unet_number=1
            )

            loss_unet2 = decoder_trainer(
                video_embed=video_embed, video=video, unet_number=2
            )

            test_losses_unet1.append(loss_unet1)
            test_losses_unet2.append(loss_unet2)

        print(
            f"Epoch {epoch}/{args.epochs} | ",
            f"avg train loss unet1: {np.mean(train_losses_unet1):.3f} | ",
            f"avg train loss unet2: {np.mean(train_losses_unet2):.3f} | ",
            f"avg test loss unet1: {np.mean(test_losses_unet1):.3f} | ",
            f"avg test loss unet2: {np.mean(test_losses_unet2):.3f} | ",
        )

        if args.use_wandb:
            performance_now = {
                "epoch": epoch,
                "train_loss_unet1": np.mean(train_losses_unet1),
                "train_loss_unet2": np.mean(train_losses_unet2),
                "test_loss_unet1": np.mean(test_losses_unet1),
                "test_loss_unet2": np.mean(test_losses_unet2),
                "lrate_unet1": getattr(decoder_trainer, "optim0").param_groups[0]["lr"],
                "lrate_unet2": getattr(decoder_trainer, "optim1").param_groups[0]["lr"],
            }
            wandb.log(performance_now)

        torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_last.pt"))

        test_loss = np.mean(test_losses_unet1) + np.mean(test_losses_unet2)
        if test_loss < min_test_loss:
            cprint(f"New best. Saving models to {run_dir}", color="cyan")
            torch.save(decoder.state_dict(), os.path.join(run_dir, "decoder_best.pt"))

            min_test_loss = test_loss


if __name__ == "__main__":
    run()
