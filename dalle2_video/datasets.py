import os, sys
import torch
from glob import glob
from natsort import natsorted
from termcolor import cprint
from tqdm import tqdm
import logging

import clip

logger = logging.getLogger("datasets")


class CelebVTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts_path: str, videos_path: str) -> None:
        """_summary_
        Args:
            texts_path: Path to the tokenized texts.
        """
        texts = torch.load(texts_path)
        print(texts.shape)
