import clip
import hydra
from omegaconf import DictConfig
import logging

from dalle2_video.datasets import CelebVTextDataset


@hydra.main(version_base=None, config_path="configs", config_name="celebv-text")
def train(args: DictConfig) -> None:
    logging.getLogger().setLevel(eval(f"logging.{args.log_level}"))

    CelebVTextDataset(args)


if __name__ == "__main__":
    train()
