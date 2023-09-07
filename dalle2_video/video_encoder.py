"""
Modified from: https://github.com/rishikksh20/ViViT-pytorch/blob/master/vivit.py
"""
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                # fmt: off
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                # fmt: on
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


# FIXME
class ViViT(nn.Module):
    def __init__(
        self,
        frame_size: int,
        patch_size: int,
        num_frames: int,
        dim: int = 192,
        depth: int = 4,
        heads: int = 3,
        in_channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        scale_dim_attn: int = 4,
        scale_dim_conv: int = 5,
    ):
        super().__init__()
        assert frame_size % patch_size == 0, "Image dimensions must be divisible by the patch size."  # fmt: skip

        num_patches = (frame_size // patch_size) ** 2
        patch_dim = in_channels * patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim)
        )
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim_attn, dropout
        )

        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim_attn, dropout
        )
        self.temporal_conv = nn.Sequential(
            Rearrange("b t d -> b d t"),
            Rearrange("b d (t s) -> b (d s) t", s=scale_dim_conv),
            nn.Conv1d(dim * scale_dim_conv, dim, kernel_size=3, padding="same"),
            Rearrange("b d (t s) -> b (d s) t", s=scale_dim_conv),
            nn.Conv1d(dim * scale_dim_conv, dim, kernel_size=3, padding="same"),
            nn.SiLU(),
        )

        self.dropout = nn.Dropout(emb_dropout)

        to_out_dim = dim * num_frames // (scale_dim_conv**2)
        self.to_out = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(to_out_dim),
            nn.Linear(to_out_dim, dim),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, _, n, _ = x.shape

        x += self.pos_embedding[:, :, :n]  # self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)

        h = self.temporal_transformer(x)
        h = self.temporal_conv(h)

        return self.to_out(h)
