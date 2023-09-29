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


@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def run(args: DictConfig) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = os.path.join("runs/celebv-text", args.train_name, "clip")

    device = f"cuda:{args.clip.cuda_id}"

    # -----------------------
    #       Dataloader
    # -----------------------
    dataset = CelebVTextDataset(
        texts_path=args.texts_dirs.tokenized,
        videos_path=args.videos_dirs.preprocessed,
    )

    loader_args = {
        "batch_size": args.clip.batch_size,
        "collate_fn": dataset.collate_fn,
        "drop_last": False,
        "num_workers": args.clip.num_workers,
        "pin_memory": True,
        "prefetch_factor": None,
    }
    loader = torch.utils.data.DataLoader(dataset=dataset, shuffle=True, **loader_args)

    # ---------------------
    #        Models
    # ---------------------
    video_encoder = ViViT(num_frames=args.seq_len * args.fps, **args.video_encoder)
    video_encoder.load_state_dict(
        torch.load(os.path.join(run_dir, "video_encoder_last.pt"))
    )
    video_encoder.to(device)

    clip_model, _ = clip.load(args.clip_model, device=device)

    # ----------------------------
    # Evaluation (save embeddings)
    # ----------------------------
    text_embeds = []
    video_embeds = []

    video_encoder.eval()
    for texts, videos in tqdm(loader, desc="Evaluation"):
        texts, videos = texts.to(device), videos.to(device)

        text_embeds.append(clip_model.encode_text(texts).float())
        video_embeds.append(video_encoder(videos))

    text_embeds = torch.cat(text_embeds)
    video_embeds = torch.cat(video_embeds)

    cprint(f"Texts (embedded): {text_embeds.shape}, Videos (embedded): {video_embeds.shape}", "cyan")  # fmt: skip

    torch.save(text_embeds, args.texts_dirs.embed)
    torch.save(video_embeds, args.videos_dirs.embed)


if __name__ == "__main__":
    run()
