import os, sys
import numpy as np
import torch
from torchvision.transforms import Compose
import cv2
from PIL import Image
import h5py
import hydra
from omegaconf import DictConfig
from glob import glob
from tqdm import tqdm
from termcolor import cprint
from typing import Optional

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


def preprocess_video(
    video_path: str, preprocess: Compose, num_frames: int
) -> Optional[np.ndarray]:
    """Preprocesses a video. Although the original videos have different number of frames,
    this function takes the first `num_frames` of a video.
    Args:
        video_path: _description_
        preprocess: Resize(size=224) -> CenterCrop(size=224) -> ToTensor() -> Normalize()
        num_frames: _description_
    Returns:
        video: ( c, t, h, w )
    """
    cap = cv2.VideoCapture(video_path)
    # frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    video = []
    for _ in range(num_frames):
        ret, frame = cap.read()

        if not ret:
            return None

        frame = Image.fromarray(frame)
        frame = preprocess(frame)

        video.append(frame.numpy())

    return np.stack(video).transpose(1, 0, 2, 3)


@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def run(args: DictConfig) -> None:
    details_paths = glob(
        os.path.join(args.texts_dirs.root, args.texts_dirs.details, "*.txt")
    )

    _, img_preproc = clip.load("ViT-B/32")

    texts = []

    num_frames = int(args.seq_len * args.fps)
    hdf = h5py.File(os.path.join(args.videos_dirs.root, "preprocessed.h5"), "w")
    videos = hdf.require_dataset(
        name="videos",
        shape=(0, 3, num_frames, 224, 224),
        maxshape=(None, 3, num_frames, 224, 224),
        dtype=np.float32,
    )

    for details_path in tqdm(details_paths, desc="Preprocessing"):
        video_path = os.path.join(
            args.videos_dirs.root,
            args.videos_dirs.untar,
            f"{os.path.splitext(os.path.basename(details_path))[0]}.mp4",
        )

        if not os.path.exists(video_path):
            cprint(f"Video {video_path} not found.", "yellow")
            continue

        video = preprocess_video(video_path, img_preproc, num_frames)
        if video is None:
            cprint(f"Video {video_path} has less than {num_frames} frames.", "yellow")
            continue

        videos.resize(videos.shape[0] + 1, axis=0)
        videos[-1] = video

        text = load_text(details_path, args.texts_dirs)
        texts.append(text)

    hdf.close()

    cprint(f"Tokenizing {len(texts)} texts.", "cyan")
    texts = clip.tokenize(texts, truncate=True)

    torch.save(texts, os.path.join(args.texts_dirs.root, "tokenized.pt"))


if __name__ == "__main__":
    run()
