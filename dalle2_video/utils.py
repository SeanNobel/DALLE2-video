import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from termcolor import cprint
from typing import Union, Optional, Callable


class CLIPLoss(nn.Module):
    def __init__(self, reduction: str = "mean", init_temperature: float = 5.0):
        super().__init__()
        self.compute_similarity = nn.CosineSimilarity(dim=-1)
        self._criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.temp = nn.Parameter(torch.tensor([float(init_temperature)]))

    def forward(self, x, y, fast=True, return_logits=False):
        batch_size = x.size(0)
        assert batch_size > 1, "Batch size must be greater than 1."
        targets = torch.arange(batch_size, requires_grad=False).long().to(device=x.device)

        if not fast:
            # less efficient way
            x_ = rearrange(x, "b f t -> 1 b (f t)")
            y_ = rearrange(y, "b f t -> b 1 (f t)")
            logits = self.compute_similarity(x_, y_)  # s

        else:
            # fast way
            x = x.reshape(batch_size, -1)
            y = y.reshape(batch_size, -1)

            # NOTE: scale the embeddings to unit norm
            x = x / x.norm(dim=-1, keepdim=True)
            y = y / y.norm(dim=-1, keepdim=True)

            # get dot products
            logits = torch.matmul(x, y.T)

        # scale by temperature (learned)
        logits *= torch.exp(self.temp)

        # NOTE: as in https://arxiv.org/abs/2103.00020
        loss = (
            self._criterion(logits, targets) + self._criterion(logits.t(), targets)
        ) / 2

        if return_logits:
            return logits, loss
        else:
            return loss


class Classifier(nn.Module):
    # NOTE: experimental

    def __init__(self, args):
        super(Classifier, self).__init__()

        # NOTE: Do we need to adjust the accuracies for the dataset size?
        self.factor = 1  # self.batch_size / 241

    @torch.no_grad()
    def forward(
        self, Z: torch.Tensor, Y: torch.Tensor, sequential=False, return_pred=False
    ) -> torch.Tensor:
        batch_size = Z.size(0)

        diags = torch.arange(batch_size).to(device=Z.device)

        Z = Z.contiguous().view(batch_size, -1)
        Y = Y.contiguous().view(batch_size, -1)

        Z = Z / Z.norm(dim=-1, keepdim=True)
        Y = Y / Y.norm(dim=-1, keepdim=True)

        # NOTE: avoid CUDA out of memory like this
        if sequential:
            similarity = torch.empty(batch_size, batch_size).to(device=Z.device)

            pbar = tqdm(total=batch_size, desc="Similarity matrix of test size")

            for i in range(batch_size):
                # similarity[i] = (Z[i] @ Y.T) / torch.clamp(
                #     (Z[i].norm() * Y.norm(dim=1)), min=1e-8
                # )
                similarity[i] = Z[i] @ Y.T

                pbar.update(1)

            similarity = similarity.T

            torch.cuda.empty_cache()

        else:
            Z = rearrange(Z, "b f -> 1 b f")
            Y = rearrange(Y, "b f -> b 1 f")
            similarity = F.cosine_similarity(Z, Y, dim=-1)  # ( B, B )

        # NOTE: max similarity of speech and M/EEG representations is expected for corresponding windows
        top1accuracy = (similarity.argmax(axis=1) == diags).to(torch.float).mean().item()
        try:
            top10accuracy = np.mean(
                [
                    label in row
                    for row, label in zip(
                        torch.topk(similarity, 10, dim=1, largest=True)[1], diags
                    )
                ]
            )
        except:
            print(similarity.size())

            raise

        if return_pred:
            cprint(similarity.argmax(axis=1).shape, "cyan")
            cprint(Y.shape, "cyan")
            return (
                top1accuracy,
                top10accuracy,
                similarity.argmax(axis=1).cpu(),
            )

        else:
            return top1accuracy, top10accuracy, similarity


def sequential_apply(
    X: Union[torch.Tensor, np.ndarray],
    # NOTE: nn.Module is a hint for general DNNs. Callable is a hint for CLIP encoder
    model: Union[transforms.Compose, nn.Module, Callable],
    batch_size: int,
    device: Optional[str] = None,
    desc: str = "",
) -> torch.Tensor:
    """Avoid CPU / CUDA out of memory.
    Args:
        X (torch.Tensor): _description_
        model (Union[transforms.Compose, VisionEncoder]): _description_
        batch_size (int): _description_
    Returns:
        torch.Tensor: _description_
    """
    # NOTE: This is for torchvision transforms, which doesn't accept a batch of samples.
    # A bit of messy implementation.
    if isinstance(model, transforms.Compose) and isinstance(X, np.ndarray):
        # NOTE: np.split needs number of subarrays, while torch.split needs the size of chunks.
        return torch.cat(
            [
                model(Image.fromarray(_X.squeeze())).unsqueeze(0)
                for _X in np.split(X, X.shape[0])
            ]
        )

    orig_device = X.device

    if device is None:
        device = orig_device

    # NOTE: sequential_apply doesn't do sequential application if batch_size == X.shape[0].
    if batch_size == X.shape[0]:
        assert isinstance(X, torch.Tensor) and isinstance(model, nn.Module)

        return model(X.to(device)).to(orig_device)

    return torch.cat(
        [
            model(_X.to(device)).to(orig_device)
            for _X in tqdm(torch.split(X, batch_size), desc=desc)
        ]
    )
