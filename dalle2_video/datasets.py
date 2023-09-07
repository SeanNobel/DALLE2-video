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
from typing import List, Optional, Tuple, Union, Any

import clip

logger = logging.getLogger("datasets")


def exists(val: Any) -> bool:
    return val is not None


class CelebVTextCollator(nn.Module):
    def __init__(self, videos_ref: h5py._hl.dataset.Dataset) -> None:
        """Loads videos from h5 reference object.
        Args:
            videos_ref: h5py reference object to the videos.
        """
        super().__init__()

        self.videos_ref = videos_ref

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor]:
        x = torch.stack([item[0] for item in batch])
        # texts -> ( b, 77 ), video_embeds -> ( b, 512 )

        # NOTE: item[2] is subject_idx and item[1] is sample_idx
        videos = np.stack([self.videos_ref[item[1]] for item in batch])
        # ( b, c, t, h, w )
        videos = torch.from_numpy(videos).permute(0, 2, 1, 3, 4)
        # ( b, t, c, h, w )

        return x, videos


class CelebVTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        texts_path: Optional[str] = None,
        videos_path: Optional[str] = None,
        text_embeds_path: Optional[str] = None,
        video_embeds_path: Optional[str] = None,
    ) -> None:
        """_summary_
        Args:
            texts_path: Path to the tokenized texts.
        """
        # fmt: off
        self.texts = torch.load(texts_path) if exists(texts_path) else None

        self.videos_ref = h5py.File(videos_path, "r")["videos"] if exists(videos_path) else None
        self.videos = torch.arange(len(self.videos_ref), dtype=torch.int64) if exists(videos_path) else None

        self.text_embeds = torch.load(text_embeds_path) if exists(text_embeds_path) else None

        self.video_embeds = torch.load(video_embeds_path) if exists(video_embeds_path) else None

        if exists(self.texts) and exists(self.videos):
            assert self.text_embeds is None and self.video_embeds is None, "Embeddings are not needed for CLIP training."
            self.stage = "CLIP"
            cprint(f"Texts (tokenized): {self.texts.shape}, Videos (ref): {self.videos_ref.shape}", "cyan")
            assert len(self.texts) == len(self.videos)

        elif exists(self.text_embeds) and exists(self.video_embeds):
            assert self.texts is None and self.videos is None, "Texts and videos are not needed for prior training."
            self.stage = "prior"
            cprint(f"Text embeds: {self.text_embeds.shape}, Video embeds: {self.video_embeds.shape}", "cyan")
            assert len(self.text_embeds) == len(self.video_embeds)

        elif exists(self.video_embeds) and exists(self.videos):
            assert self.texts is None and self.text_embeds is None, "Texts and text embeddings are not needed for decoder training."
            self.stage = "decoder"
            cprint(f"Video embeds: {self.video_embeds.shape}, Videos (ref): {self.videos_ref.shape}", "cyan")
            assert len(self.video_embeds) == len(self.videos)

        else:
            raise ValueError("No matching training stage for the given paths.")

        if exists(self.videos):
            self.collate_fn = CelebVTextCollator(self.videos_ref)
        else:
            self.collate_fn = None

        # fmt: on

    def __len__(self):
        if self.stage == "CLIP":
            return len(self.texts)
        else:
            return len(self.video_embeds)

    def __getitem__(self, i: int):
        if self.stage == "CLIP":
            return self.texts[i], self.videos[i]
        elif self.stage == "prior":
            return self.text_embeds[i], self.video_embeds[i]
        elif self.stage == "decoder":
            return self.video_embeds[i], self.videos[i]
