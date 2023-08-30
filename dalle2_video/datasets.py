import os, sys
import numpy as np
import torch
import torch.nn as nn
import h5py
from glob import glob
from natsort import natsorted
from termcolor import cprint
from tqdm import tqdm
import logging
from typing import List, Optional, Tuple, Union

import clip

logger = logging.getLogger("datasets")


class CelebVTextCollator(nn.Module):
    def __init__(self, videos_ref: h5py._hl.dataset.Dataset) -> None:
        """Loads videos from h5 reference object.
        Args:
            videos_ref: h5py reference object to the videos.
        """
        super().__init__()

        self.videos_ref = videos_ref

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        texts = torch.stack([item[0] for item in batch])  # ( b, 77 )

        # NOTE: item[2] is subject_idx and item[1] is sample_idx
        videos = np.stack([self.videos_ref[item[1]] for item in batch])
        # ( b, c, t, h, w )
        videos = torch.from_numpy(videos)

        return texts, videos


class CelebVTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts_path: str, videos_path: str) -> None:
        """_summary_
        Args:
            texts_path: Path to the tokenized texts.
        """
        self.texts = torch.load(texts_path)

        self.videos_ref = h5py.File(videos_path, "r")["videos"]
        self.videos = torch.arange(len(self.videos_ref), dtype=torch.int64)

        cprint(f"Texts (tokenized): {self.texts.shape}, Videos (ref): {self.videos_ref.shape}", "cyan")  # fmt: skip
        assert len(self.texts) == len(self.videos), "Texts and videos don't have the same number of samples."  # fmt: skip

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i: int):
        return self.texts[i], self.videos[i]
