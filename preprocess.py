import os, sys
import numpy as np
import torch
from torchvision.transforms import Compose
import cv2
from PIL import Image
import hydra
from omegaconf import DictConfig
from glob import glob
from tqdm import tqdm
from termcolor import cprint
from typing import Tuple, List

import clip


def load_text(details_path: str, texts_dirs: DictConfig):
    """Loads text files of details, action, emotion, light direction, light intensity,
    and light temperature as far as they exist, and concatenate them into one text.
    Args:
        details_path: _description_
        texts_dirs (DictConfig): _description_
    Returns:
        text: Text description of a video.
    """
    text = []

    with open(details_path, "r") as f:
        # NOTE: Each text file contains many lines but each line basically says the same thing.
        text.append(f.readline()[:-1])  # remove \n

    for additional in texts_dirs.additional.values():
        additional_path = os.path.join(
            texts_dirs.root, additional, os.path.basename(details_path)
        )
        try:
            with open(additional_path, "r") as f:
                text.append(f.readline()[:-1])
        except FileNotFoundError:
            cprint(f"File {additional_path} not found.", "yellow")

    return " ".join(text)


def preprocess_and_save_video(video_path: str, preprocess: Compose) -> None:
    """Preprocesses a video. And
    Args:
        video_path (str): _description_
    Returns:
        video (torch.Tensor): _description_
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    video = []
    pbar = tqdm(total=frame_count, desc="Preprocessing video")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # frame = Image.fromarray(frame)
        # frame = preprocess(frame)

        video.append(frame)

        pbar.update(1)

    # video = torch.stack(video)
    video = np.stack(video)
    print(video.shape)


@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def run(args: DictConfig) -> None:
    details_paths = glob(
        os.path.join(args.texts_dirs.root, args.texts_dirs.details, "*.txt")
    )

    _, preprocess = clip.load("ViT-B/32")

    texts = []
    for details_path in tqdm(details_paths, desc="Preprocessing"):
        video_path = os.path.join(
            args.videos_dir, f"{os.path.splitext(os.path.basename(details_path))[0]}.mp4"
        )

        if not os.path.exists(video_path):
            cprint(f"Video {video_path} not found.", "yellow")
            continue

        text = load_text(details_path, args.texts_dirs)
        texts.append(text)

        preprocess_and_save_video(video_path, preprocess)

    cprint(f"Tokenizing {len(texts)} texts.", "cyan")
    texts = clip.tokenize(texts, truncate=True)

    torch.save(texts, os.path.join(args.texts_dirs.root, "tokenized.pt"))


if __name__ == "__main__":
    run()
