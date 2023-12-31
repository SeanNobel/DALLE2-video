import sys
from typing import Any, Callable, Optional, Union, Tuple, List, Dict
from termcolor import cprint
import logging

import torch
import torch.nn as nn

from x_clip import CLIP
from coca_pytorch import CoCa

# from dalle2_video._dalle2_pytorch import *
from dalle2_pytorch.dalle2_pytorch import *
from dalle2_pytorch.vqgan_vae import NullVQGanVAE, VQGanVAE

logger = logging.getLogger("dalle2_video")


def Downsample3D(dim, dim_out=None):
    # https://arxiv.org/abs/2208.03641 shows this is the most optimal way to downsample
    # named SP-conv in the paper, but basically a pixel unshuffle
    dim_out = default(dim_out, dim)
    return nn.Sequential(
        Rearrange("b c t (h s1) (w s2) -> b (c s1 s2) t h w", s1=2, s2=2),
        nn.Conv3d(dim * 4, dim_out, 1),
    )


def NearestUpsample3D(dim, dim_out=None):
    dim_out = default(dim_out, dim)

    return nn.Sequential(
        nn.Upsample(scale_factor=(1, 2, 2), mode="nearest"),
        nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
    )


class PixelShuffleUpsample3D(nn.Module):
    """
    code shared by @MalumaDev at DALLE2-pytorch for addressing checkboard artifacts
    https://arxiv.org/ftp/arxiv/papers/1707/1707.02937.pdf
    """

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.conv = nn.Conv3d(dim, dim_out * 4, 1)
        self.act = nn.SiLU()

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        self.init_conv_(self.conv)

    def init_conv_(self, conv):
        o, i, t, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, t, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x: torch.Tensor):
        """_summary_
        Args:
            x ( b, c, t, h, w ): _description_
        Returns:
            _type_: _description_
        """
        x = self.act(self.conv(x))

        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")

        x = self.pixel_shuffle(x)

        return rearrange(x, "(b t) c h w -> b c t h w", t=t)


def temporal_apply(
    fn: Callable,
    x: torch.Tensor,
    *args,
    **kwargs,
):
    """Apply any function sequentially to each frame of a video.
    Args:
        x ( b, c, t, h, w ): _description_
    Returns:
        _type_: _description_
    """
    return torch.stack(
        [fn(x[:, :, _t], *args, **kwargs) for _t in range(x.shape[2])],
        dim=2,
    )


class Block3D(nn.Module):
    def __init__(self, dim, dim_out, groups=8, weight_standardization=False):
        super().__init__()

        # TODO: Implement WeightStandardizedConv3d
        # conv_klass = nn.Conv2d if not weight_standardization else WeightStandardizedConv2d

        # NOTE: "we change each 3x3 convolution into a 1x3x3 convolution" in https://arxiv.org/abs/2204.03458
        self.project = nn.Conv3d(dim, dim_out, kernel_size=(1, 3, 3), padding=(0, 1, 1))

        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """_summary_
        Args:
            x ( b, c, t, h, w ): _description_
            scale_shift (( b, dim_out, 1, 1, 1 ), ( b, dim_out, 1, 1, 1 )): _description_
        Returns:
            _type_: _description_
        """
        x = self.project(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        *,
        cond_dim=None,
        time_cond_dim=None,
        groups=8,
        weight_standardization=False,
        cosine_sim_cross_attn=False,
    ):
        super().__init__()

        self.time_mlp = None

        if exists(time_cond_dim):
            self.time_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_cond_dim, dim_out * 2)
            )

        self.cross_attn = None

        if exists(cond_dim):
            self.cross_attn = CrossAttention(
                dim=dim_out, context_dim=cond_dim, cosine_sim=cosine_sim_cross_attn
            )

        self.block1 = Block3D(
            dim, dim_out, groups=groups, weight_standardization=weight_standardization
        )
        self.block2 = Block3D(
            dim_out, dim_out, groups=groups, weight_standardization=weight_standardization
        )
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None, cond=None
    ):
        """_summary_
        Args:
            x ( b, c, t, h, w ): _description_
            time_emb ( b, time_cond_dim ): _description_. Defaults to None.
            cond (_type_, optional): _description_. Defaults to None.
        Returns:
            _type_: _description_
        """
        scale_shift = None
        if exists(self.time_mlp) and exists(time_emb):
            time_emb = self.time_mlp(time_emb)  # ( b, dim_out * 2 )
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)
            # ( b, dim_out, 1, 1, 1 ), ( b, dim_out, 1, 1, 1 )

        h = self.block1(x, scale_shift=scale_shift)

        if exists(self.cross_attn):
            assert exists(cond)

            h = rearrange(h, "b c ... -> b ... c")
            h, ps = pack([h], "b * c")

            h = self.cross_attn(h, context=cond) + h

            (h,) = unpack(h, ps, "b * c")
            h = rearrange(h, "b ... c -> b c ...")

        h = self.block2(h)

        return h + self.res_conv(x)


class CrossEmbedLayer3D(nn.Module):
    def __init__(self, dim_in, kernel_sizes, dim_out=None, stride=2):
        super().__init__()
        assert all([*map(lambda t: (t % 2) == (stride % 2), kernel_sizes)])
        dim_out = default(dim_out, dim_in)

        # FIXME: Doing ksize=1 conv for time dimension.
        kernel_sizes = [(1, ksize, ksize) for ksize in sorted(kernel_sizes)]
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(dim_out / (2**i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, dim_out - sum(dim_scales)]

        self.convs = nn.ModuleList([])
        for ksize, dim_scale in zip(kernel_sizes, dim_scales):
            self.convs.append(
                nn.Conv3d(
                    dim_in,
                    dim_scale,
                    ksize,
                    stride=(1, stride, stride),
                    padding=(0, (ksize[1] - stride) // 2, (ksize[2] - stride) // 2),
                )
            )

    def forward(self, x):
        """_summary_
        Args:
            x ( b, c=3, t, h, w ): _description_
        Returns:
            x: ( b, c', t, h, w ): Initial convoluted video.
        """
        fmaps = tuple(map(lambda conv: conv(x), self.convs))
        x = torch.cat(fmaps, dim=1)

        return x


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        *,
        video_embed_dim=None,
        text_embed_dim=None,
        cond_dim=None,
        num_image_tokens=4,
        num_time_tokens=2,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        channels_out=None,
        self_attn=False,
        attn_dim_head=32,
        attn_heads=16,
        lowres_cond=False,  # for cascading diffusion - https://cascaded-diffusion.github.io/
        lowres_noise_cond=False,  # for conditioning on low resolution noising, based on Imagen
        self_cond=False,  # set this to True to use the self-conditioning technique from - https://arxiv.org/abs/2208.04202
        sparse_attn=False,
        cosine_sim_cross_attn=False,
        cosine_sim_self_attn=False,
        attend_at_middle=True,  # whether to have a layer of attention at the bottleneck (can turn off for higher resolution in cascading DDPM, before bringing in efficient attention)
        cond_on_text_encodings=False,
        max_text_len=256,
        cond_on_video_embeds=False,
        add_video_embeds_to_time=True,  # alerted by @mhh0318 to a phrase in the paper - "Specifically, we modify the architecture described in Nichol et al. (2021) by projecting and adding CLIP embeddings to the existing timestep embedding"
        init_dim=None,
        init_conv_ksize=7,  # (1, 7, 7),  # NOTE: space-only 3D convolution.
        resnet_groups=8,
        resnet_weight_standardization=False,
        num_resnet_blocks=2,
        init_cross_embed=True,  # False, # FIXME: implement CrossEmbedLayer3D later.
        init_cross_embed_kernel_sizes=(3, 7, 15),
        cross_embed_downsample=False,
        cross_embed_downsample_kernel_sizes=(2, 4),
        memory_efficient=False,
        scale_skip_connection=False,
        pixel_shuffle_upsample=True,
        final_conv_ksize=1,
        combine_upsample_fmaps=False,  # whether to combine the outputs of all upsample blocks, as in unet squared paper
        checkpoint_during_training=False,
        **kwargs,
    ):
        super().__init__()
        # save locals to take care of some hyperparameters for cascading DDPM

        self._locals = locals()
        del self._locals["self"]
        del self._locals["__class__"]

        # for eventual cascading diffusion

        self.lowres_cond = lowres_cond

        # whether to do self conditioning

        self.self_cond = self_cond

        # determine dimensions

        self.channels = channels
        self.channels_out = default(channels_out, channels)

        # initial number of channels depends on
        # (1) low resolution conditioning from cascading ddpm paper, conditioned on previous unet output in the cascade
        # (2) self conditioning (bit diffusion paper)

        init_channels = channels * (1 + int(lowres_cond) + int(self_cond))

        init_dim = default(init_dim, dim)

        self.init_conv = (
            CrossEmbedLayer3D(
                init_channels,
                dim_out=init_dim,
                kernel_sizes=init_cross_embed_kernel_sizes,
                stride=1,
            )
            if init_cross_embed
            else nn.Conv3d(
                in_channels=init_channels,
                out_channels=init_dim,
                kernel_size=(1, init_conv_ksize, init_conv_ksize),
                padding=(0, init_conv_ksize // 2, init_conv_ksize // 2),
            )
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # e.g. [128, 128, 256, 512, 1024]
        in_out = list(zip(dims[:-1], dims[1:]))
        # e.g. [(128, 128), (128, 256), (256, 512), (512, 1024)]

        num_stages = len(in_out)

        # time, image embeddings, and optional text encoding

        cond_dim = default(cond_dim, dim)
        time_cond_dim = dim * 4

        self.to_time_hiddens = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_cond_dim), nn.GELU()
        )

        self.to_time_tokens = nn.Sequential(
            nn.Linear(time_cond_dim, cond_dim * num_time_tokens),
            Rearrange("b (r d) -> b r d", r=num_time_tokens),
        )

        self.to_time_cond = nn.Sequential(nn.Linear(time_cond_dim, time_cond_dim))

        self.video_to_tokens = (
            nn.Sequential(
                nn.Linear(video_embed_dim, cond_dim * num_image_tokens),
                Rearrange("b (n d) -> b n d", n=num_image_tokens),
            )
            if cond_on_video_embeds and video_embed_dim != cond_dim
            else nn.Identity()
        )

        self.to_video_hiddens = (
            nn.Sequential(nn.Linear(video_embed_dim, time_cond_dim), nn.GELU())
            if cond_on_video_embeds and add_video_embeds_to_time
            else None
        )

        self.norm_cond = nn.LayerNorm(cond_dim)
        self.norm_mid_cond = nn.LayerNorm(cond_dim)

        # text encoding conditioning (optional)

        self.text_to_cond = None
        self.text_embed_dim = None

        if cond_on_text_encodings:
            assert exists(
                text_embed_dim
            ), "text_embed_dim must be given to the unet if cond_on_text_encodings is True"
            self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
            self.text_embed_dim = text_embed_dim

        # low resolution noise conditiong, based on Imagen's upsampler training technique

        self.lowres_noise_cond = lowres_noise_cond

        self.to_lowres_noise_cond = (
            nn.Sequential(
                SinusoidalPosEmb(dim),
                nn.Linear(dim, time_cond_dim),
                nn.GELU(),
                nn.Linear(time_cond_dim, time_cond_dim),
            )
            if lowres_noise_cond
            else None
        )

        # finer control over whether to condition on image embeddings and text encodings
        # so one can have the latter unets in the cascading DDPMs only focus on super-resoluting

        self.cond_on_text_encodings = cond_on_text_encodings
        self.cond_on_video_embeds = cond_on_video_embeds

        # for classifier free guidance

        self.null_video_embed = nn.Parameter(torch.randn(1, num_image_tokens, cond_dim))
        self.null_video_hiddens = nn.Parameter(torch.randn(1, time_cond_dim))

        self.max_text_len = max_text_len
        self.null_text_embed = nn.Parameter(torch.randn(1, max_text_len, cond_dim))

        # whether to scale skip connection, adopted in Imagen

        self.skip_connect_scale = 1.0 if not scale_skip_connection else (2**-0.5)

        # attention related params

        attn_kwargs = dict(
            heads=attn_heads, dim_head=attn_dim_head, cosine_sim=cosine_sim_self_attn
        )

        self_attn = cast_tuple(self_attn, num_stages)

        create_self_attn = lambda dim: RearrangeToSequence(
            Residual(Attention(dim, **attn_kwargs))
        )

        # resnet block klass

        resnet_groups = cast_tuple(resnet_groups, num_stages)
        top_level_resnet_group = first(resnet_groups)

        num_resnet_blocks = cast_tuple(num_resnet_blocks, num_stages)

        # downsample klass

        if cross_embed_downsample:
            downsample_klass = partial(
                CrossEmbedLayer3D, kernel_sizes=cross_embed_downsample_kernel_sizes
            )
        else:
            downsample_klass = Downsample3D

        # upsample klass

        if pixel_shuffle_upsample:
            # NOTE: Just applying PixelShuffleUpsample frame-wise.
            upsample_klass = PixelShuffleUpsample3D
        else:
            upsample_klass = NearestUpsample3D

        # prepare resnet klass

        resnet_block = partial(
            ResnetBlock3D,
            cosine_sim_cross_attn=cosine_sim_cross_attn,
            weight_standardization=resnet_weight_standardization,
        )

        # give memory efficient unet an initial resnet block

        self.init_resnet_block = (
            resnet_block(
                init_dim,
                init_dim,
                time_cond_dim=time_cond_dim,
                groups=top_level_resnet_group,
            )
            if memory_efficient
            else None
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        skip_connect_dims = []  # keeping track of skip connection dimensions
        upsample_combiner_dims = []
        # keeping track of dimensions for final upsample feature map combiner

        for ind, (
            (dim_in, dim_out),
            groups,
            layer_num_resnet_blocks,
            layer_self_attn,
        ) in enumerate(zip(in_out, resnet_groups, num_resnet_blocks, self_attn)):
            is_first = ind == 0
            is_last = ind >= (num_resolutions - 1)
            layer_cond_dim = cond_dim if not is_first else None

            dim_layer = dim_out if memory_efficient else dim_in
            skip_connect_dims.append(dim_layer)

            # TODO: Implement attention for 3D
            if layer_self_attn:
                attention = create_self_attn(dim_layer)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_layer, **attn_kwargs))
            else:
                attention = nn.Identity()

            self.downs.append(
                nn.ModuleList(
                    [
                        downsample_klass(dim_in, dim_out=dim_out)
                        if memory_efficient
                        else None,
                        resnet_block(
                            dim_layer,
                            dim_layer,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                resnet_block(
                                    dim_layer,
                                    dim_layer,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        downsample_klass(dim_layer, dim_out=dim_out)
                        if not is_last and not memory_efficient
                        else nn.Conv3d(dim_layer, dim_out, 1),
                    ]
                )
            )

        mid_dim = dims[-1]

        self.mid_block1 = resnet_block(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )
        self.mid_attn = create_self_attn(mid_dim)
        self.mid_block2 = resnet_block(
            mid_dim,
            mid_dim,
            cond_dim=cond_dim,
            time_cond_dim=time_cond_dim,
            groups=resnet_groups[-1],
        )

        for ind, (
            (dim_in, dim_out),
            groups,
            layer_num_resnet_blocks,
            layer_self_attn,
        ) in enumerate(
            zip(
                reversed(in_out),
                reversed(resnet_groups),
                reversed(num_resnet_blocks),
                reversed(self_attn),
            )
        ):
            is_last = ind >= (len(in_out) - 1)
            layer_cond_dim = cond_dim if not is_last else None

            skip_connect_dim = skip_connect_dims.pop()

            if layer_self_attn:
                attention = create_self_attn(dim_out)
            elif sparse_attn:
                attention = Residual(LinearAttention(dim_out, **attn_kwargs))
            else:
                attention = nn.Identity()

            upsample_combiner_dims.append(dim_out)

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(
                            dim_out + skip_connect_dim,
                            dim_out,
                            cond_dim=layer_cond_dim,
                            time_cond_dim=time_cond_dim,
                            groups=groups,
                        ),
                        nn.ModuleList(
                            [
                                resnet_block(
                                    dim_out + skip_connect_dim,
                                    dim_out,
                                    cond_dim=layer_cond_dim,
                                    time_cond_dim=time_cond_dim,
                                    groups=groups,
                                )
                                for _ in range(layer_num_resnet_blocks)
                            ]
                        ),
                        attention,
                        upsample_klass(dim_out, dim_in)
                        if not is_last or memory_efficient
                        else nn.Identity(),
                    ]
                )
            )

        # whether to combine outputs from all upsample blocks for final resnet block

        self.upsample_combiner = UpsampleCombiner(
            dim=dim,
            enabled=combine_upsample_fmaps,
            dim_ins=upsample_combiner_dims,
            dim_outs=(dim,) * len(upsample_combiner_dims),
        )

        # a final resnet block

        self.final_resnet_block = resnet_block(
            self.upsample_combiner.dim_out + dim,
            dim,
            time_cond_dim=time_cond_dim,
            groups=top_level_resnet_group,
        )

        out_dim_in = dim + (channels if lowres_cond else 0)

        self.to_out = nn.Conv3d(
            out_dim_in,
            self.channels_out,
            kernel_size=(1, final_conv_ksize, final_conv_ksize),
            padding=(0, final_conv_ksize // 2, final_conv_ksize // 2),
        )

        zero_init_(self.to_out)  # since both OpenAI and @crowsonkb are doing it

        # whether to checkpoint during training

        self.checkpoint_during_training = checkpoint_during_training

    # if the current settings for the unet are not correct
    # for cascading DDPM, then reinit the unet with the right settings
    def cast_model_parameters(
        self,
        *,
        lowres_cond,
        lowres_noise_cond,
        channels,
        channels_out,
        cond_on_image_embeds,
        cond_on_text_encodings,
    ):
        if (
            lowres_cond == self.lowres_cond
            and channels == self.channels
            and cond_on_image_embeds == self.cond_on_video_embeds
            and cond_on_text_encodings == self.cond_on_text_encodings
            and lowres_noise_cond == self.lowres_noise_cond
            and channels_out == self.channels_out
        ):
            return self

        updated_kwargs = dict(
            lowres_cond=lowres_cond,
            channels=channels,
            channels_out=channels_out,
            cond_on_image_embeds=cond_on_image_embeds,
            cond_on_text_encodings=cond_on_text_encodings,
            lowres_noise_cond=lowres_noise_cond,
        )

        return self.__class__(**{**self._locals, **updated_kwargs})

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, text_cond_drop_prob=1.0, video_cond_drop_prob=1.0, **kwargs
        )
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        *,
        video_embed: torch.Tensor,
        lowres_cond_video=None,
        lowres_noise_level=None,
        text_encodings=None,
        video_cond_drop_prob=0.0,
        text_cond_drop_prob=0.0,
        blur_sigma=None,
        blur_kernel_size=None,
        disable_checkpoint=False,
        self_cond=None,
    ):
        """_summary_
        Args:
            x ( b, c, t, h, w ): _description_
            time ( b, ): _description_
            video_embed (_type_): _description_
            lowres_cond_video (_type_, optional): _description_. Defaults to None.
            lowres_noise_level (_type_, optional): _description_. Defaults to None.
            text_encodings (_type_, optional): _description_. Defaults to None.
            video_cond_drop_prob (float, optional): _description_. Defaults to 0.0.
            text_cond_drop_prob (float, optional): _description_. Defaults to 0.0.
            blur_sigma (_type_, optional): _description_. Defaults to None.
            blur_kernel_size (_type_, optional): _description_. Defaults to None.
            disable_checkpoint (bool, optional): _description_. Defaults to False.
            self_cond (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        batch_size, device = x.shape[0], x.device

        # add low resolution conditioning, if present

        assert not (
            self.lowres_cond and not exists(lowres_cond_video)
        ), "low resolution conditioning image must be present"

        # concat self conditioning, if needed

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x, self_cond), dim=1)

        # concat low resolution conditioning

        if exists(lowres_cond_video):
            # NOTE: Concatenation for channels
            x = torch.cat((x, lowres_cond_video), dim=1)

        # initial convolution

        x = self.init_conv(x)  # ( b, c, t, h, w )
        r = x.clone()  # final residual

        # time conditioning (same as image diffusion)

        time = time.type_as(x)  # ( b, )
        time_hiddens = self.to_time_hiddens(time)  # ( b, time_cond_dim )

        time_tokens = self.to_time_tokens(
            time_hiddens
        )  # ( b, num_time_tokens, cond_dim )
        t = self.to_time_cond(time_hiddens)  # ( b, time_cond_dim )

        # low res noise conditioning (similar to time above)

        if exists(lowres_noise_level):
            assert exists(self.to_lowres_noise_cond), "lowres_noise_cond must be set to True on instantiation of the unet in order to conditiong on lowres noise"  # fmt: skip
            lowres_noise_level = lowres_noise_level.type_as(x)
            t = t + self.to_lowres_noise_cond(lowres_noise_level)

        # conditional dropout

        video_keep_mask = prob_mask_like(
            (batch_size,), 1 - video_cond_drop_prob, device=device
        )  # ( b, )

        text_keep_mask = prob_mask_like(
            (batch_size,), 1 - text_cond_drop_prob, device=device
        )  # ( b, )
        text_keep_mask = rearrange(text_keep_mask, "b -> b 1 1")  # ( b, 1, 1 )

        # video embedding to be summed to time embedding
        # discovered by @mhh0318 in the paper

        if exists(video_embed) and exists(self.to_video_hiddens):
            video_hiddens = self.to_video_hiddens(video_embed)
            video_keep_mask_hidden = rearrange(video_keep_mask, "b -> b 1")
            null_video_hiddens = self.null_video_hiddens.to(video_hiddens.dtype)

            # mask hiddens with some probability
            video_hiddens = torch.where(
                video_keep_mask_hidden, video_hiddens, null_video_hiddens
            )

            t = t + video_hiddens

        # mask out image embedding depending on condition dropout
        # for classifier free guidance

        video_tokens = None

        if self.cond_on_video_embeds:
            video_keep_mask_embed = rearrange(video_keep_mask, "b -> b 1 1")
            video_tokens = self.video_to_tokens(video_embed)
            null_video_embed = self.null_video_embed.to(video_tokens.dtype)
            # for some reason pytorch AMP not working

            video_tokens = torch.where(
                video_keep_mask_embed, video_tokens, null_video_embed
            )

        # take care of text encodings (optional)

        text_tokens = None

        if exists(text_encodings) and self.cond_on_text_encodings:
            assert (
                text_encodings.shape[0] == batch_size
            ), f"the text encodings being passed into the unet does not have the proper batch size - text encoding shape {text_encodings.shape} - required batch size is {batch_size}"
            assert (
                self.text_embed_dim == text_encodings.shape[-1]
            ), f"the text encodings you are passing in have a dimension of {text_encodings.shape[-1]}, but the unet was created with text_embed_dim of {self.text_embed_dim}."

            text_mask = torch.any(text_encodings != 0.0, dim=-1)

            text_tokens = self.text_to_cond(text_encodings)

            text_tokens = text_tokens[:, : self.max_text_len]
            text_mask = text_mask[:, : self.max_text_len]

            text_tokens_len = text_tokens.shape[1]
            remainder = self.max_text_len - text_tokens_len

            if remainder > 0:
                text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
                text_mask = F.pad(text_mask, (0, remainder), value=False)

            text_mask = rearrange(text_mask, "b n -> b n 1")

            assert (
                text_mask.shape[0] == text_keep_mask.shape[0]
            ), f"text_mask has shape of {text_mask.shape} while text_keep_mask has shape {text_keep_mask.shape}. text encoding is of shape {text_encodings.shape}"
            text_keep_mask = text_mask & text_keep_mask

            null_text_embed = self.null_text_embed.to(
                text_tokens.dtype
            )  # for some reason pytorch AMP not working

            text_tokens = torch.where(text_keep_mask, text_tokens, null_text_embed)

        # main conditioning tokens (c)

        c = time_tokens  # ( b, num_time_tokens, cond_dim )

        if exists(video_tokens):
            c = torch.cat((c, video_tokens), dim=-2)

        # text and video conditioning tokens (mid_c)
        # to save on compute, only do cross attention based conditioning on the inner most layers of the Unet

        mid_c = c if not exists(text_tokens) else torch.cat((c, text_tokens), dim=-2)

        # normalize conditioning tokens

        c = self.norm_cond(c)
        mid_c = self.norm_mid_cond(mid_c)

        # gradient checkpointing

        can_checkpoint = (
            self.training and self.checkpoint_during_training and not disable_checkpoint
        )
        apply_checkpoint_fn = make_checkpointable if can_checkpoint else identity

        # make checkpointable modules

        init_resnet_block, mid_block1, mid_attn, mid_block2, final_resnet_block = [
            maybe(apply_checkpoint_fn)(module)
            for module in (
                self.init_resnet_block,
                self.mid_block1,
                self.mid_attn,
                self.mid_block2,
                self.final_resnet_block,
            )
        ]

        can_checkpoint_cond = lambda m: isinstance(m, ResnetBlock)
        downs, ups = [
            maybe(apply_checkpoint_fn)(m, condition=can_checkpoint_cond)
            for m in (self.downs, self.ups)
        ]

        # initial resnet block

        if exists(init_resnet_block):
            x = init_resnet_block(x, t)

        # go through the layers of the unet, down and up

        down_hiddens = []
        up_hiddens = []

        for pre_downsample, init_block, resnet_blocks, attn, post_downsample in downs:
            if exists(pre_downsample):
                x = pre_downsample(x)

            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = resnet_block(x, t, c)
                down_hiddens.append(x.contiguous())

            x = attn(x)
            down_hiddens.append(x.contiguous())

            if exists(post_downsample):
                x = post_downsample(x)

        x = mid_block1(x, t, mid_c)

        if exists(mid_attn):
            x = mid_attn(x)

        x = mid_block2(x, t, mid_c)

        connect_skip = lambda fmap: torch.cat(
            (fmap, down_hiddens.pop() * self.skip_connect_scale), dim=1
        )

        for init_block, resnet_blocks, attn, upsample in ups:
            x = connect_skip(x)
            x = init_block(x, t, c)

            for resnet_block in resnet_blocks:
                x = connect_skip(x)
                x = resnet_block(x, t, c)

            x = attn(x)

            up_hiddens.append(x.contiguous())
            x = upsample(x)

        x = self.upsample_combiner(x, up_hiddens)

        x = torch.cat((x, r), dim=1)

        x = final_resnet_block(x, t)

        if exists(lowres_cond_video):
            x = torch.cat((x, lowres_cond_video), dim=1)

        return self.to_out(x)


class UnetTemporalConv(Unet):
    """
    NOTE: This is very preliminary implementation and probably doesn't work.
    TODO: Implement Unet3D
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.learned_var = self.channels_out == self.channels * 2

        self.temporal_conv = nn.Conv3d(
            in_channels=self.channels_out,
            out_channels=self.channels_out,
            kernel_size=(3, 1, 1),
            stride=1,
            padding="same",
        )

    def forward_with_cond_scale(self, *args, cond_scale=1.0, **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(
            *args, text_cond_drop_prob=1.0, video_cond_drop_prob=1.0, **kwargs
        )

        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        video_embed: torch.Tensor,
        lowres_cond_video: Optional[torch.Tensor],
        video_cond_drop_prob: float = 0.0,
        **kwargs,
    ):
        """_summary_
        Args:
            x ( b, t, c, h, w ): _description_
            video_embed ( b, d, t ): _description_
            lowres_cond_video ( b, t, c, h_low, w_low ): _description_
            video_cond_drop_prob (float): _description_
        Returns:
            _type_: _description_
        """
        b, t, c, _, _ = x.shape

        x = x.view(b * t, *x.shape[2:])
        video_embed = video_embed.contiguous().view(b * t, -1)

        if exists(lowres_cond_video):
            lowres_cond_video = lowres_cond_video.view(
                b * t, *lowres_cond_video.shape[2:]
            )

        times = torch.repeat_interleave(times, repeats=t, dim=0)

        x = super().forward(
            x,
            times,
            image_embed=video_embed,
            lowres_cond_img=lowres_cond_video,
            image_cond_drop_prob=video_cond_drop_prob,
            **kwargs,
        )

        # ( b * t, c * 2, h, w )
        # cprint(f"{x.shape}, {self.learned_var}, {self.channels_out}", "red")

        # FIXME: Getting rid of learned variance here.
        # if self.learned_var:
        #     x = x.chunk(2, dim=1)[0]

        x = x.view(b, t, *x.shape[1:]).permute(0, 2, 1, 3, 4)
        # ( b, c, t, h, w )

        # else:
        #     x = x.view(b, t, c * 2, *x.shape[2:]).permute(0, 2, 1, 3, 4)
        #     # ( b, c * 2, t, h, w )

        x = self.temporal_conv(x)

        return x.permute(0, 2, 1, 3, 4)


class LowresVideoConditioner(nn.Module):
    def __init__(
        self,
        downsample_first=True,
        use_blur=True,
        blur_prob=0.5,
        blur_sigma=0.6,
        blur_kernel_size=3,
        use_noise=False,
        input_video_range=None,
        normalize_video_fn=identity,
        unnormalize_video_fn=identity,
    ):
        super().__init__()
        self.downsample_first = downsample_first
        self.input_video_range = input_video_range

        self.use_blur = use_blur
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.blur_kernel_size = blur_kernel_size

        self.use_noise = use_noise
        self.normalize_video = normalize_video_fn
        self.unnormalize_video = unnormalize_video_fn
        self.noise_scheduler = (
            NoiseScheduler(beta_schedule="linear", timesteps=1000, loss_type="l2")
            if use_noise
            else None
        )

    def noise_video(self, cond_fmap, noise_levels=None):
        assert exists(self.noise_scheduler)

        batch = cond_fmap.shape[0]
        cond_fmap = self.normalize_video(cond_fmap)

        random_noise_levels = default(
            noise_levels, lambda: self.noise_scheduler.sample_random_times(batch)
        )
        cond_fmap = self.noise_scheduler.q_sample(
            cond_fmap, t=random_noise_levels, noise=torch.randn_like(cond_fmap)
        )

        cond_fmap = self.unnormalize_video(cond_fmap)
        return cond_fmap, random_noise_levels

    @staticmethod
    def blur_image(cond_fmap: torch.Tensor, blur_sigma, blur_kernel_size):
        # NOTE: Currently this method is applied to each frame independently.
        # when training, blur the low resolution conditional image
        # allow for drawing a random sigma between lo and hi float values

        if isinstance(blur_sigma, tuple):
            blur_sigma = tuple(map(float, blur_sigma))
            blur_sigma = random.uniform(*blur_sigma)

        # allow for drawing a random kernel size between lo and hi int values

        if isinstance(blur_kernel_size, tuple):
            blur_kernel_size = tuple(map(int, blur_kernel_size))
            kernel_size_lo, kernel_size_hi = blur_kernel_size
            blur_kernel_size = random.randrange(kernel_size_lo, kernel_size_hi + 1)

        cond_fmap = gaussian_blur2d(
            cond_fmap, cast_tuple(blur_kernel_size, 2), cast_tuple(blur_sigma, 2)
        )

        return cond_fmap

    def forward(
        self,
        cond_fmap: torch.Tensor,
        *,
        target_frame_size,
        downsample_frame_size: Optional[int] = None,
        target_frame_number: Optional[int] = None,
        downsample_frame_number: Optional[int] = None,
        should_blur=True,
        blur_sigma=None,
        blur_kernel_size=None,
    ):
        if self.downsample_first and exists(downsample_frame_size):
            cond_fmap = temporal_apply(
                resize_image_to,
                cond_fmap,
                downsample_frame_size,
                clamp_range=self.input_video_range,
                nearest=True,
            )

        # blur is only applied 50% of the time
        # section 3.1 in https://arxiv.org/abs/2106.15282

        if self.use_blur and should_blur and random.random() < self.blur_prob:
            cond_fmap = temporal_apply(
                self.blur_image,
                cond_fmap,
                default(blur_sigma, self.blur_sigma),
                default(blur_kernel_size, self.blur_kernel_size),
            )

        # resize to target image size

        cond_fmap = temporal_apply(
            resize_image_to,
            cond_fmap,
            target_frame_size,
            clamp_range=self.input_video_range,
            nearest=True,
        )

        # noise conditioning, as done in Imagen
        # as a replacement for the BSR noising, and potentially replace blurring for first stage too

        random_noise_levels = None

        if self.use_noise:
            cond_fmap, random_noise_levels = self.noise_video(cond_fmap)

        # return conditioning feature map, as well as the augmentation noise levels

        return cond_fmap, random_noise_levels


class VideoDecoder(nn.Module):
    def __init__(
        self,
        unet: Union[Unet3D, UnetTemporalConv],
        *,
        clip=None,
        frame_size: Optional[int] = None,
        channels=3,
        vae=tuple(),
        timesteps=1000,
        sample_timesteps=None,
        video_cond_drop_prob=0.1,
        text_cond_drop_prob=0.5,
        loss_type="l2",
        beta_schedule=None,
        predict_x_start=False,
        predict_v=False,
        predict_x_start_for_latent_diffusion=False,
        frame_sizes: Optional[Tuple[int]] = None,
        frame_numbers: Optional[Tuple[int]] = None,
        random_crop_sizes=None,  # whether to random crop the image at that stage in the cascade (super resoluting convolutions at the end may be able to generalize on smaller crops)
        use_noise_for_lowres_cond=False,  # whether to use Imagen-like noising for low resolution conditioning
        use_blur_for_lowres_cond=True,  # whether to use the blur conditioning used in the original cascading ddpm paper, as well as DALL-E2
        lowres_downsample_first=True,  # cascading ddpm - resizes to lower resolution, then to next conditional resolution + blur
        blur_prob=0.5,  # cascading ddpm - when training, the gaussian blur is only applied 50% of the time
        blur_sigma=0.6,  # cascading ddpm - blur sigma
        blur_kernel_size=3,  # cascading ddpm - blur kernel size
        lowres_noise_sample_level=0.2,  # in imagen paper, they use a 0.2 noise level at sample time for low resolution conditioning
        clip_denoised=True,
        clip_x_start=True,
        clip_adapter_overrides=dict(),
        learned_variance=True,
        learned_variance_constrain_frac=False,
        vb_loss_weight=0.001,
        unconditional=False,  # set to True for generating images without conditioning
        auto_normalize_video: bool = True,  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.95,
        p2_loss_weight_gamma=0.0,  # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time - 1. is recommended
        p2_loss_weight_k=1,
        ddim_sampling_eta=0.0,  # can be set to 0. for deterministic sampling afaict
    ):
        """_summary_
        Args:
            unet (_type_): _description_
            clip (_type_, optional): _description_. Defaults to None.
            frame_size (_type_, optional): _description_. Defaults to None.
            channels (int, optional): _description_. Defaults to 3.
            vae (_type_, optional): _description_. Defaults to tuple().
            timesteps (int, optional): _description_. Defaults to 1000.
            sample_timesteps (_type_, optional): _description_. Defaults to None.
            video_cond_drop_prob (float, optional): _description_. Defaults to 0.1.
            text_cond_drop_prob (float, optional): _description_. Defaults to 0.5.
            loss_type (str, optional): _description_. Defaults to "l2".
            beta_schedule (_type_, optional): _description_. Defaults to None.
            predict_x_start (bool, optional): _description_. Defaults to False.
            predict_v (bool, optional): _description_. Defaults to False.
            predict_x_start_for_latent_diffusion (bool, optional): _description_. Defaults to False.
            frame_sizes: For cascading ddpm, frame size at each stage.
            frame_numbers: For cascading temporal super-resolution.
            clip_x_start (bool, optional): _description_. Defaults to True.
            clip_adapter_overrides (_type_, optional): _description_. Defaults to dict().
            learned_variance (bool, optional): _description_. Defaults to True.
            learned_variance_constrain_frac (bool, optional): _description_. Defaults to False.
            vb_loss_weight (float, optional): _description_. Defaults to 0.001.
            unconditional (bool, optional): _description_. Defaults to False.
            p2_loss_weight_gamma (float, optional): _description_. Defaults to 0.0.
            fromhttps (arxiv.org, optional): _description_. Defaults to 1.
            ddim_sampling_eta (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__()

        # clip

        self.clip = None
        if exists(clip):
            assert not unconditional, "clip must not be given if doing unconditional image training"  # fmt: skip
            assert channels == clip.image_channels, f"channels of image ({channels}) should be equal to the channels that CLIP accepts ({clip.image_channels})"  # fmt: skip

            if isinstance(clip, CLIP):
                clip = XClipAdapter(clip, **clip_adapter_overrides)
            elif isinstance(clip, CoCa):
                clip = CoCaAdapter(clip, **clip_adapter_overrides)

            freeze_model_and_make_eval_(clip)
            assert isinstance(clip, BaseClipAdapter)

            self.clip = clip

        # determine image size, with image_size and image_sizes taking precedence

        if exists(frame_size) or exists(frame_sizes):
            assert exists(frame_size) ^ exists(frame_sizes), "only one of image_size or image_sizes must be given"  # fmt: skip
            frame_size = default(frame_size, lambda: frame_sizes[-1])

        elif exists(clip):
            frame_size = clip.image_size

        else:
            raise Exception("either image_size, image_sizes, or clip must be given to decoder")  # fmt: skip

        # channels

        self.channels = channels

        # normalize and unnormalize image functions

        self.normalize_video = (
            normalize_neg_one_to_one if auto_normalize_video else identity
        )
        self.unnormalize_video = (
            unnormalize_zero_to_one if auto_normalize_video else identity
        )

        # verify conditioning method

        unets = cast_tuple(unet)
        num_unets = len(unets)
        self.num_unets = num_unets

        self.unconditional = unconditional

        # automatically take care of ensuring that first unet is unconditional
        # while the rest of the unets are conditioned on the low resolution shorter video produced by previous unet

        vaes = pad_tuple_to_length(
            cast_tuple(vae), len(unets), fillvalue=NullVQGanVAE(channels=self.channels)
        )

        # whether to use learned variance, defaults to True for the first unet in the cascade, as in paper

        learned_variance = pad_tuple_to_length(
            cast_tuple(learned_variance), len(unets), fillvalue=False
        )
        self.learned_variance = learned_variance
        self.learned_variance_constrain_frac = learned_variance_constrain_frac  # whether to constrain the output of the network (the interpolation fraction) from 0 to 1
        self.vb_loss_weight = vb_loss_weight

        # default and validate conditioning parameters

        use_noise_for_lowres_cond = cast_tuple(
            use_noise_for_lowres_cond, num_unets - 1, validate=False
        )
        use_blur_for_lowres_cond = cast_tuple(
            use_blur_for_lowres_cond, num_unets - 1, validate=False
        )

        if len(use_noise_for_lowres_cond) < num_unets:
            use_noise_for_lowres_cond = (False, *use_noise_for_lowres_cond)

        if len(use_blur_for_lowres_cond) < num_unets:
            use_blur_for_lowres_cond = (False, *use_blur_for_lowres_cond)

        assert not use_noise_for_lowres_cond[0], "first unet will never need low res noise conditioning"  # fmt: skip
        assert not use_blur_for_lowres_cond[0], "first unet will never need low res blur conditioning"  # fmt: skip

        assert num_unets == 1 or all((use_noise or use_blur) for use_noise, use_blur in zip(use_noise_for_lowres_cond[1:], use_blur_for_lowres_cond[1:]))  # fmt: skip

        # construct unets and vaes

        self.unets = nn.ModuleList([])
        self.vaes = nn.ModuleList([])

        for ind, (
            one_unet,
            one_vae,
            one_unet_learned_var,
            lowres_noise_cond,
        ) in enumerate(zip(unets, vaes, learned_variance, use_noise_for_lowres_cond)):
            assert isinstance(one_unet, UnetTemporalConv) or isinstance(one_unet, Unet3D)
            assert isinstance(one_vae, (VQGanVAE, NullVQGanVAE))

            is_first = ind == 0
            latent_dim = one_vae.encoded_dim if exists(one_vae) else None

            unet_channels = default(latent_dim, self.channels)
            unet_channels_out = unet_channels * (1 if not one_unet_learned_var else 2)

            one_unet = one_unet.cast_model_parameters(
                lowres_cond=not is_first,
                lowres_noise_cond=lowres_noise_cond,
                cond_on_image_embeds=not unconditional and is_first,
                cond_on_text_encodings=not unconditional
                and one_unet.cond_on_text_encodings,
                channels=unet_channels,
                channels_out=unet_channels_out,
            )

            self.unets.append(one_unet)
            self.vaes.append(one_vae.copy_for_eval())

        # sampling timesteps, defaults to non-ddim with full timesteps sampling

        self.sample_timesteps = cast_tuple(sample_timesteps, num_unets)
        self.ddim_sampling_eta = ddim_sampling_eta

        # create noise schedulers per unet

        if not exists(beta_schedule):
            beta_schedule = (
                "cosine",
                *(("cosine",) * max(num_unets - 2, 0)),
                *(("linear",) * int(num_unets > 1)),
            )

        beta_schedule = cast_tuple(beta_schedule, num_unets)
        p2_loss_weight_gamma = cast_tuple(p2_loss_weight_gamma, num_unets)

        self.noise_schedulers = nn.ModuleList([])

        for ind, (
            unet_beta_schedule,
            unet_p2_loss_weight_gamma,
            sample_timesteps,
        ) in enumerate(zip(beta_schedule, p2_loss_weight_gamma, self.sample_timesteps)):
            assert (
                not exists(sample_timesteps) or sample_timesteps <= timesteps
            ), f"sampling timesteps {sample_timesteps} must be less than or equal to the number of training timesteps {timesteps} for unet {ind + 1}"

            noise_scheduler = NoiseScheduler(
                beta_schedule=unet_beta_schedule,
                timesteps=timesteps,
                loss_type=loss_type,
                p2_loss_weight_gamma=unet_p2_loss_weight_gamma,
                p2_loss_weight_k=p2_loss_weight_k,
            )

            self.noise_schedulers.append(noise_scheduler)

        # unet frame sizes

        # NOTE: Currently only supporting square videos
        frame_sizes = default(frame_sizes, (frame_size,))
        frame_sizes = tuple(sorted(set(frame_sizes)))

        assert self.num_unets == len(frame_sizes), f"you did not supply the correct number of u-nets ({self.num_unets}) for resolutions {frame_sizes}"  # fmt: skip
        self.frame_sizes = frame_sizes
        self.sample_channels = cast_tuple(self.channels, len(frame_sizes))

        # unet frame numbers

        self.frame_numbers = frame_numbers

        # random crop sizes (for super-resoluting unets at the end of cascade?)

        self.random_crop_sizes = cast_tuple(random_crop_sizes, len(frame_sizes))
        assert not exists(self.random_crop_sizes[0]), "you would not need to randomly crop the image for the base unet"  # fmt: skip

        # predict x0 config

        self.predict_x_start = (
            cast_tuple(predict_x_start, len(unets))
            if not predict_x_start_for_latent_diffusion
            else tuple(map(lambda t: isinstance(t, VQGanVAE), self.vaes))
        )

        # predict v

        self.predict_v = cast_tuple(predict_v, len(unets))

        # input image range

        self.input_video_range = (-1.0 if not auto_normalize_video else 0.0, 1.0)

        # cascading ddpm related stuff

        lowres_conditions = tuple(map(lambda t: t.lowres_cond, self.unets))
        assert lowres_conditions == (False, *((True,) * (num_unets - 1)),), "the first unet must be unconditioned (by low resolution image), and the rest of the unets must have `lowres_cond` set to True"  # fmt: skip

        self.lowres_conds = nn.ModuleList([])

        for unet_index, use_noise, use_blur in zip(
            range(num_unets), use_noise_for_lowres_cond, use_blur_for_lowres_cond
        ):
            if unet_index == 0:
                self.lowres_conds.append(None)
                continue

            lowres_cond = LowresVideoConditioner(
                downsample_first=lowres_downsample_first,
                use_blur=use_blur,
                use_noise=use_noise,
                blur_prob=blur_prob,
                blur_sigma=blur_sigma,
                blur_kernel_size=blur_kernel_size,
                input_video_range=self.input_video_range,
                normalize_video_fn=self.normalize_video,
                unnormalize_video_fn=self.unnormalize_video,
            )

            self.lowres_conds.append(lowres_cond)

        self.lowres_noise_sample_level = lowres_noise_sample_level

        # classifier free guidance

        self.video_cond_drop_prob = video_cond_drop_prob
        self.text_cond_drop_prob = text_cond_drop_prob
        self.can_classifier_guidance = (
            video_cond_drop_prob > 0.0 or text_cond_drop_prob > 0.0
        )

        # whether to clip when sampling

        self.clip_denoised = clip_denoised
        self.clip_x_start = clip_x_start

        # dynamic thresholding settings, if clipping denoised during sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # device tracker

        self.register_buffer("_dummy", torch.Tensor([True]), persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def condition_on_text_encodings(self):
        return any(
            [unet.cond_on_text_encodings for unet in self.unets if isinstance(unet, Unet)]
        )

    def get_unet(self, unet_number):
        assert 0 < unet_number <= self.num_unets
        index = unet_number - 1
        return self.unets[index]

    def parse_unet_output(self, learned_variance, output):
        var_interp_frac_unnormalized = None

        if learned_variance:
            output, var_interp_frac_unnormalized = output.chunk(2, dim=2)

        return UnetOutput(output, var_interp_frac_unnormalized)

    @contextmanager
    def one_unet_in_gpu(self, unet_number=None, unet=None, cuda="cuda"):
        assert exists(unet_number) ^ exists(unet)

        if exists(unet_number):
            unet = self.get_unet(unet_number)

        # devices

        cuda, cpu = torch.device(cuda), torch.device("cpu")

        self.to(cuda)

        devices = [module_device(unet) for unet in self.unets]

        self.unets.to(cpu)
        unet.to(cuda)

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    def dynamic_threshold(self, x):
        """proposed in https://arxiv.org/abs/2205.11487 as an improved clamping in the setting of classifier free guidance"""

        # s is the threshold amount
        # static thresholding would just be s = 1
        s = 1.0
        if self.use_dynamic_thres:
            s = torch.quantile(
                rearrange(x, "b ... -> b (...)").abs(),
                self.dynamic_thres_percentile,
                dim=-1,
            )

            s.clamp_(min=1.0)
            s = s.view(-1, *((1,) * (x.ndim - 1)))

        # clip by threshold, depending on whether static or dynamic
        x = x.clamp(-s, s) / s
        return x

    def p_mean_variance(
        self,
        unet,
        x,
        t,
        video_embed,
        noise_scheduler,
        text_encodings=None,
        lowres_cond_vid=None,
        self_cond=None,
        clip_denoised=True,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        cond_scale=1.0,
        model_output=None,
        lowres_noise_level=None,
    ):
        assert not (cond_scale != 1.0 and not self.can_classifier_guidance), "the decoder was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)"  # fmt: skip

        model_output = default(
            model_output,
            lambda: unet.forward_with_cond_scale(
                x,
                t,
                video_embed=video_embed,
                text_encodings=text_encodings,
                cond_scale=cond_scale,
                lowres_cond_video=lowres_cond_vid,
                self_cond=self_cond,
                lowres_noise_level=lowres_noise_level,
            ),
        )

        pred, var_interp_frac_unnormalized = self.parse_unet_output(
            learned_variance, model_output
        )

        # v-prediction: https://arxiv.org/pdf/2202.00512.pdf
        if predict_v:
            x_start = noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif predict_x_start:
            x_start = pred
        else:
            x_start = noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)

        if clip_denoised:
            x_start = self.dynamic_threshold(x_start)

        model_mean, posterior_variance, posterior_log_variance = noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)  # fmt: skip

        if learned_variance:
            # if learned variance, posterio variance and posterior log variance are predicted by the network
            # by an interpolation of the max and min log beta values
            # eq 15 - https://arxiv.org/abs/2102.09672
            min_log = extract(noise_scheduler.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(torch.log(noise_scheduler.betas), t, x.shape)
            var_interp_frac = unnormalize_zero_to_one(var_interp_frac_unnormalized)

            if self.learned_variance_constrain_frac:
                var_interp_frac = var_interp_frac.sigmoid()

            posterior_log_variance = (
                var_interp_frac * max_log + (1 - var_interp_frac) * min_log
            )
            posterior_variance = posterior_log_variance.exp()

        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(
        self,
        unet,
        x,
        t,
        video_embed,
        noise_scheduler,
        text_encodings=None,
        cond_scale=1.0,
        lowres_cond_vid=None,
        self_cond=None,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_noise_level=None,
    ):
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            unet,
            x=x,
            t=t,
            video_embed=video_embed,
            text_encodings=text_encodings,
            cond_scale=cond_scale,
            lowres_cond_vid=lowres_cond_vid,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            predict_x_start=predict_x_start,
            predict_v=predict_v,
            noise_scheduler=noise_scheduler,
            learned_variance=learned_variance,
            lowres_noise_level=lowres_noise_level,
        )

        noise = torch.randn_like(x)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(
        self,
        unet,
        shape,
        video_embed,
        noise_scheduler,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_cond_vid=None,
        text_encodings=None,
        cond_scale=1,
        is_latent_diffusion=False,
        lowres_noise_level=None,
        # inpaint_image=None,
        # inpaint_mask=None,
        # inpaint_resample_times=5,
    ):
        device = self.device

        b = shape[0]
        vid = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        # is_inpaint = exists(inpaint_image)
        resample_times = 1  # inpaint_resample_times if is_inpaint else 1

        # NOTE: Inpainting is not supported for video.
        # if is_inpaint:
        #     inpaint_image = self.normalize_img(inpaint_image)
        #     inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
        #     inpaint_mask = rearrange(inpaint_mask, "b h w -> b 1 h w").float()
        #     inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
        #     inpaint_mask = inpaint_mask.bool()

        if not is_latent_diffusion:
            lowres_cond_vid = maybe(self.normalize_video)(lowres_cond_vid)

        for time in tqdm(
            reversed(range(0, noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=noise_scheduler.num_timesteps,
        ):
            is_last_timestep = time == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                times = torch.full((b,), time, device=device, dtype=torch.long)

                # if is_inpaint:
                #     # following the repaint paper
                #     # https://arxiv.org/abs/2201.09865
                #     noised_inpaint_image = noise_scheduler.q_sample(
                #         inpaint_image, t=times
                #     )
                #     img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                vid, x_start = self.p_sample(
                    unet,
                    vid,
                    times,
                    video_embed=video_embed,
                    text_encodings=text_encodings,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_vid=lowres_cond_vid,
                    lowres_noise_level=lowres_noise_level,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    noise_scheduler=noise_scheduler,
                    learned_variance=learned_variance,
                    clip_denoised=clip_denoised,
                )

                # if is_inpaint and not (is_last_timestep or is_last_resample_step):
                #     # in repaint, you renoise and resample up to 10 times every step
                #     img = noise_scheduler.q_sample_from_to(img, times - 1, times)

        # if is_inpaint:
        #     img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        unnormalize_vid = self.unnormalize_video(vid)

        return unnormalize_vid

    @torch.no_grad()
    def p_sample_loop_ddim(
        self,
        unet,
        shape,
        image_embed,
        noise_scheduler,
        timesteps,
        eta=1.0,
        predict_x_start=False,
        predict_v=False,
        learned_variance=False,
        clip_denoised=True,
        lowres_cond_img=None,
        text_encodings=None,
        cond_scale=1,
        is_latent_diffusion=False,
        lowres_noise_level=None,
        inpaint_image=None,
        inpaint_mask=None,
        inpaint_resample_times=5,
    ):
        batch, device, total_timesteps, alphas, eta = (
            shape[0],
            self.device,
            noise_scheduler.num_timesteps,
            noise_scheduler.alphas_cumprod,
            self.ddim_sampling_eta,
        )

        times = torch.linspace(0.0, total_timesteps, steps=timesteps + 2)[:-1]

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        time_pairs = list(filter(lambda t: t[0] > t[1], time_pairs))

        is_inpaint = exists(inpaint_image)
        resample_times = inpaint_resample_times if is_inpaint else 1

        # NOTE: Inpainting is not supported for video.
        # if is_inpaint:
        #     inpaint_image = self.normalize_img(inpaint_image)
        #     inpaint_image = resize_image_to(inpaint_image, shape[-1], nearest=True)
        #     inpaint_mask = rearrange(inpaint_mask, "b h w -> b 1 h w").float()
        #     inpaint_mask = resize_image_to(inpaint_mask, shape[-1], nearest=True)
        #     inpaint_mask = inpaint_mask.bool()

        img = torch.randn(shape, device=device)

        x_start = None  # for self-conditioning

        if not is_latent_diffusion:
            lowres_cond_img = maybe(self.normalize_video)(lowres_cond_img)

        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            is_last_timestep = time_next == 0

            for r in reversed(range(0, resample_times)):
                is_last_resample_step = r == 0

                alpha = alphas[time]
                alpha_next = alphas[time_next]

                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)

                if is_inpaint:
                    # following the repaint paper
                    # https://arxiv.org/abs/2201.09865
                    noised_inpaint_image = noise_scheduler.q_sample(
                        inpaint_image, t=time_cond
                    )
                    img = (img * ~inpaint_mask) + (noised_inpaint_image * inpaint_mask)

                self_cond = x_start if unet.self_cond else None

                unet_output = unet.forward_with_cond_scale(
                    img,
                    time_cond,
                    image_embed=image_embed,
                    text_encodings=text_encodings,
                    cond_scale=cond_scale,
                    self_cond=self_cond,
                    lowres_cond_img=lowres_cond_img,
                    lowres_noise_level=lowres_noise_level,
                )

                pred, _ = self.parse_unet_output(learned_variance, unet_output)

                # predict x0

                if predict_v:
                    x_start = noise_scheduler.predict_start_from_v(
                        img, t=time_cond, v=pred
                    )
                elif predict_x_start:
                    x_start = pred
                else:
                    x_start = noise_scheduler.predict_start_from_noise(
                        img, t=time_cond, noise=pred
                    )

                # maybe clip x0

                if clip_denoised:
                    x_start = self.dynamic_threshold(x_start)

                # predict noise

                pred_noise = noise_scheduler.predict_noise_from_start(
                    img, t=time_cond, x0=x_start
                )

                c1 = (
                    eta
                    * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                )
                c2 = ((1 - alpha_next) - torch.square(c1)).sqrt()
                noise = torch.randn_like(img) if not is_last_timestep else 0.0

                img = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise

                if is_inpaint and not (is_last_timestep or is_last_resample_step):
                    # in repaint, you renoise and resample up to 10 times every step
                    time_next_cond = torch.full(
                        (batch,), time_next, device=device, dtype=torch.long
                    )
                    img = noise_scheduler.q_sample_from_to(img, time_next_cond, time_cond)

        if exists(inpaint_image):
            img = (img * ~inpaint_mask) + (inpaint_image * inpaint_mask)

        img = self.unnormalize_video(img)
        return img

    @torch.no_grad()
    def p_sample_loop(self, *args, noise_scheduler, timesteps=None, **kwargs):
        num_timesteps = noise_scheduler.num_timesteps

        timesteps = default(timesteps, num_timesteps)
        assert timesteps <= num_timesteps
        is_ddim = timesteps < num_timesteps

        if not is_ddim:
            return self.p_sample_loop_ddpm(
                *args, noise_scheduler=noise_scheduler, **kwargs
            )

        return self.p_sample_loop_ddim(
            *args, noise_scheduler=noise_scheduler, timesteps=timesteps, **kwargs
        )

    def p_losses(
        self,
        unet: Union[Unet3D, UnetTemporalConv],
        x_start: torch.Tensor,
        times: torch.Tensor,
        *,
        video_embed,
        noise_scheduler,
        lowres_cond_video=None,
        text_encodings=None,
        predict_x_start=False,
        predict_v=False,
        noise=None,
        learned_variance=False,
        clip_denoised=False,
        is_latent_diffusion=False,
        lowres_noise_level=None,
    ):
        """_summary_
        Args:
            unet (Union[Unet3D, UnetTemporalConv]): _description_
            x_start (torch.Tensor): _description_
            times ( batch_size, ): Random integers to sample inputs from.
            video_embed (_type_): _description_
            noise_scheduler (_type_): _description_
            lowres_cond_video (_type_, optional): _description_. Defaults to None.
            text_encodings (_type_, optional): _description_. Defaults to None.
            predict_x_start (bool, optional): _description_. Defaults to False.
            predict_v (bool, optional): _description_. Defaults to False.
            noise (_type_, optional): _description_. Defaults to None.
            learned_variance (bool, optional): _description_. Defaults to False.
            clip_denoised (bool, optional): _description_. Defaults to False.
            is_latent_diffusion (bool, optional): _description_. Defaults to False.
            lowres_noise_level (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        # normalize to [-1, 1]

        if not is_latent_diffusion:
            x_start = self.normalize_video(x_start)
            lowres_cond_video = maybe(self.normalize_video)(lowres_cond_video)

        # get x_t

        x_noisy = noise_scheduler.q_sample(x_start=x_start, t=times, noise=noise)

        # unet kwargs

        unet_kwargs = dict(
            video_embed=video_embed,
            text_encodings=text_encodings,
            lowres_cond_video=lowres_cond_video,
            lowres_noise_level=lowres_noise_level,
        )

        # self conditioning

        self_cond = None

        if unet.self_cond and random.random() < 0.5:
            with torch.no_grad():
                unet_output = unet(x_noisy, times, **unet_kwargs)
                self_cond, _ = self.parse_unet_output(learned_variance, unet_output)
                self_cond = self_cond.detach()

        # forward to get model prediction

        unet_output = unet(
            x_noisy,
            times,
            **unet_kwargs,
            self_cond=self_cond,
            video_cond_drop_prob=self.video_cond_drop_prob,
            text_cond_drop_prob=self.text_cond_drop_prob,
        )

        pred, _ = self.parse_unet_output(learned_variance, unet_output)

        if predict_v:
            target = noise_scheduler.calculate_v(x_start, times, noise)
        elif predict_x_start:
            target = x_start
        else:
            target = noise

        loss = noise_scheduler.loss_fn(pred, target, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")

        loss = noise_scheduler.p2_reweigh_loss(loss, times)

        loss = loss.mean()

        if not learned_variance:
            # return simple loss if not using learned variance
            return loss

        # most of the code below is transcribed from
        # https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
        # the Improved DDPM paper then further modified it so that the mean is detached (shown a couple lines before), and weighted to be smaller than the l1 or l2 "simple" loss
        # it is questionable whether this is really needed, looking at some of the figures in the paper, but may as well stay faithful to their implementation

        # if learning the variance, also include the extra weight kl loss

        true_mean, _, true_log_variance_clipped = noise_scheduler.q_posterior(
            x_start=x_start, x_t=x_noisy, t=times
        )
        model_mean, _, model_log_variance, _ = self.p_mean_variance(
            unet,
            x=x_noisy,
            t=times,
            video_embed=video_embed,
            noise_scheduler=noise_scheduler,
            clip_denoised=clip_denoised,
            learned_variance=True,
            model_output=unet_output,
        )

        # kl loss with detached model predicted mean, for stability reasons as in paper

        detached_model_mean = model_mean.detach()

        kl = normal_kl(
            true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance
        )
        kl = meanflat(kl) * NAT

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        decoder_nll = meanflat(decoder_nll) * NAT

        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))

        vb_losses = torch.where(times == 0, decoder_nll, kl)

        # weight the vb loss smaller, for stability, as in the paper (recommended 0.001)

        vb_loss = vb_losses.mean() * self.vb_loss_weight

        return loss + vb_loss

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        video=None,
        video_embed=None,
        text=None,
        text_encodings=None,
        batch_size=1,
        cond_scale=1.0,
        start_at_unet_number=1,
        stop_at_unet_number=None,
        distributed=False,
        # inpaint_image=None,
        # inpaint_mask=None,
        # inpaint_resample_times=5,
        one_unet_in_gpu_at_time=True,
        cuda="cuda",
    ):
        assert self.unconditional or exists(video_embed), "image embed must be present on sampling from decoder unless if trained unconditionally"  # fmt: skip

        if not self.unconditional:
            batch_size = video_embed.shape[0]

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip)
            _, text_encodings = self.clip.embed_text(text)

        assert not (self.condition_on_text_encodings and not exists(text_encodings)), "text or text encodings must be passed into decoder if specified"  # fmt: skip
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), "decoder specified not to be conditioned on text, yet it is presented"  # fmt: skip

        # assert not (exists(inpaint_image) ^ exists(inpaint_mask)), "inpaint_image and inpaint_mask (boolean mask of [batch, height, width]) must be both given for inpainting"

        vid = None
        if start_at_unet_number > 1:
            # Then we are not generating the first image and one must have been passed in
            assert exists(video), "image must be passed in if starting at unet number > 1"
            assert (video.shape[0] == batch_size), "image must have batch size of {} if starting at unet number > 1".format(batch_size)  # fmt: skip
            prev_unet_output_size = self.frame_sizes[start_at_unet_number - 2]
            vid = temporal_apply(
                resize_image_to, video, prev_unet_output_size, nearest=True
            )

        is_cuda = next(self.parameters()).is_cuda

        num_unets = self.num_unets
        cond_scale = cast_tuple(cond_scale, num_unets)

        # fmt: off
        for (
            unet_number, unet, vae, channel,
            frame_size, frame_number, predict_x_start, predict_v,
            learned_variance, noise_scheduler, lowres_cond, sample_timesteps, unet_cond_scale,
        ) in tqdm(
            zip(
                range(1, num_unets + 1), self.unets, self.vaes, self.sample_channels,
                self.frame_sizes, self.frame_numbers, self.predict_x_start, self.predict_v,
                self.learned_variance, self.noise_schedulers, self.lowres_conds, self.sample_timesteps, cond_scale,
            )
        ):
        # fmt: on
            if unet_number < start_at_unet_number:
                continue  # It's the easiest way to do it

            context = (
                self.one_unet_in_gpu(unet=unet, cuda=cuda)
                if is_cuda and one_unet_in_gpu_at_time
                else null_context()
            )

            with context:
                # prepare low resolution conditioning for upsamplers

                lowres_cond_vid = lowres_noise_level = None
                shape = (batch_size, channel, frame_number, frame_size, frame_size)
                
                if unet.lowres_cond:
                    lowres_cond_vid = temporal_apply(
                        resize_image_to,
                        vid,
                        frame_size,
                        clamp_range=self.input_video_range,
                        nearest=True,
                    )

                    if lowres_cond.use_noise:
                        lowres_noise_level = torch.full(
                            (batch_size,),
                            int(self.lowres_noise_sample_level * 1000),
                            dtype=torch.long,
                            device=self.device,
                        )
                        lowres_cond_vid, _ = lowres_cond.noise_video(
                            lowres_cond_vid, lowres_noise_level
                        )

                # latent diffusion

                is_latent_diffusion = isinstance(vae, VQGanVAE)
                frame_size = vae.get_encoded_fmap_size(frame_size)
                shape = (batch_size, vae.encoded_dim, frame_number, frame_size, frame_size)

                lowres_cond_vid = maybe(vae.encode)(lowres_cond_vid)

                # denoising loop for image

                vid = self.p_sample_loop(
                    unet,
                    shape,
                    video_embed=video_embed,
                    text_encodings=text_encodings,
                    cond_scale=unet_cond_scale,
                    predict_x_start=predict_x_start,
                    predict_v=predict_v,
                    learned_variance=learned_variance,
                    clip_denoised=not is_latent_diffusion,
                    lowres_cond_vid=lowres_cond_vid,
                    lowres_noise_level=lowres_noise_level,
                    is_latent_diffusion=is_latent_diffusion,
                    noise_scheduler=noise_scheduler,
                    timesteps=sample_timesteps,
                    # inpaint_image=inpaint_image,
                    # inpaint_mask=inpaint_mask,
                    # inpaint_resample_times=inpaint_resample_times,
                )

                vid = temporal_apply(vae.decode, vid)
                # vid = vae.decode(vid.view(-1, *vid.shape[2:]))
                # vid = vid.view(batch_size, frame_number, *vid.shape[1:])

            if exists(stop_at_unet_number) and stop_at_unet_number == unet_number:
                break

        return vid

    def forward(
        self,
        video: torch.Tensor,
        video_embed: torch.Tensor,
        text=None,
        text_encodings=None,
        unet_number=None,
        return_lowres_cond_video=False,  # whether to return the low resolution conditioning images, for debugging upsampler purposes
    ):
        """Given a unet and video, runs denoising process once.
        Args:
            video: ( batch_size, channels, frames, height, width )
            video_embed (_type_, optional): _description_. Defaults to None.
            text (_type_, optional): _description_. Defaults to None.
            text_encodings (_type_, optional): _description_. Defaults to None.
            unet_number (_type_, optional): _description_. Defaults to None.
            return_lowres_cond_video (_type_, optional): _description_. Defaults to False
        Returns:
            _type_: _description_
        """
        assert not (self.num_unets > 1 and not exists(unet_number)), f"you must specify which unet you want trained, from a range of 1 to {self.num_unets}, if you are training cascading DDPM (multiple unets)"  # fmt: skip
        unet_number = default(unet_number, 1)
        unet_index = unet_number - 1

        unet = self.get_unet(unet_number)

        vae = self.vaes[unet_index]
        noise_scheduler = self.noise_schedulers[unet_index]
        lowres_conditioner = self.lowres_conds[unet_index]
        target_frame_size = self.frame_sizes[unet_index]
        target_frame_number = self.frame_numbers[unet_index]
        predict_x_start = self.predict_x_start[unet_index]
        predict_v = self.predict_v[unet_index]
        random_crop_size = self.random_crop_sizes[unet_index]
        learned_variance = self.learned_variance[unet_index]
        b, c, t, h, w, device = *video.shape, video.device

        assert video.shape[1] == self.channels
        assert h >= target_frame_size and w >= target_frame_size

        # Random timesteps to sample from.
        times = torch.randint(
            0, noise_scheduler.num_timesteps, (b,), device=device, dtype=torch.long
        )

        # NOTE: CLIP video embedding is not supported.
        # if not exists(image_embed) and not self.unconditional:
        #     assert exists(self.clip), 'if you want to derive CLIP video embeddings automatically, you must supply `clip` to the decoder on init'
        #     image_embed, _ = self.clip.embed_image(image)

        if exists(text) and not exists(text_encodings) and not self.unconditional:
            assert exists(self.clip), "if you are passing in raw text, you need to supply `clip` to the decoder"  # fmt: skip
            _, text_encodings = self.clip.embed_text(text)

        assert not self.condition_on_text_encodings and not exists(text_encodings), "text or text encodings must be passed into decoder if specified"  # fmt: skip
        assert not (not self.condition_on_text_encodings and exists(text_encodings)), "decoder specified not to be conditioned on text, yet it is presented"  # fmt: skip

        lowres_cond_video, lowres_noise_level = (
            lowres_conditioner(
                video,
                target_frame_size=target_frame_size,
                downsample_frame_size=self.frame_sizes[unet_index - 1],
                target_frame_number=target_frame_number,
                downsample_frame_number=self.frame_numbers[unet_index - 1],
            )
            if exists(lowres_conditioner)
            else (None, None)
        )
        # TODO: Check if the original utility function can be used here.
        video = temporal_apply(resize_image_to, video, target_frame_size, nearest=True)

        if exists(random_crop_size):
            aug = K.RandomCrop((random_crop_size, random_crop_size), p=1.0)

            # make sure low res conditioner and image both get augmented the same way
            # detailed https://kornia.readthedocs.io/en/latest/augmentation.module.html?highlight=randomcrop#kornia.augmentation.RandomCrop
            video = aug(video)
            lowres_cond_video = aug(lowres_cond_video, params=aug._params)

        # NOTE: NullVQGanVAE does nothing, meaning we're doing pixel space diffusion.
        is_latent_diffusion = not isinstance(vae, NullVQGanVAE)

        vae.eval()
        with torch.no_grad():
            # NOTE: VAE doesn't do temporal information mixing.
            video = temporal_apply(vae.encode, video)

            if exists(lowres_cond_video):
                lowres_cond_video = temporal_apply(vae.encode, lowres_cond_video)

        logger.debug(f"video for Unet {unet_number}: {video.shape}")
        logger.debug(f"lowres_cond_video for Unet {unet_number}: {lowres_cond_video.shape if exists(lowres_cond_video) else None}")  # fmt: skip

        losses = self.p_losses(
            unet,
            video,
            times,
            video_embed=video_embed,
            text_encodings=text_encodings,
            lowres_cond_video=lowres_cond_video,
            predict_x_start=predict_x_start,
            predict_v=predict_v,
            learned_variance=learned_variance,
            is_latent_diffusion=is_latent_diffusion,
            noise_scheduler=noise_scheduler,
            lowres_noise_level=lowres_noise_level,
        )

        if not return_lowres_cond_video:
            return losses

        return losses, lowres_cond_video


class DALLE2Video(nn.Module):
    def __init__(
        self,
        *,
        prior,
        decoder,
        temporal_emb: bool = False,
        prior_num_samples=2,
        decoder_cuda="cuda",
    ) -> None:
        super().__init__()
        assert isinstance(prior, DiffusionPrior)
        assert isinstance(decoder, VideoDecoder)
        self.prior = prior
        self.decoder = decoder

        self.temporal_emb = temporal_emb

        self.prior_num_samples = prior_num_samples
        self.decoder_need_text_cond = self.decoder.condition_on_text_encodings

        self.decoder_cuda = decoder_cuda

    @torch.no_grad()
    @eval_decorator
    def forward(
        self,
        text: Optional[torch.Tensor] = None,
        text_embed: Optional[torch.Tensor] = None,
        cond_scale: float = 1.0,
        prior_cond_scale: float = 1.0,
    ):
        # device = module_device(self)
        # one_text = isinstance(text, str) or (not is_list_str(text) and text.shape[0] == 1)

        # if isinstance(text, str) or is_list_str(text):
        #     text = [text] if not isinstance(text, (list, tuple)) else text
        #     text = tokenizer.tokenize(text).to(device)

        if self.temporal_emb:
            b, d, t = text_embed.shape
            text_embed = text_embed.permute(0, 2, 1).reshape(-1, d)
        else:
            b, d = text_embed.shape

        video_embed = self.prior.sample(
            text_embed,
            num_samples_per_batch=self.prior_num_samples,
            cond_scale=prior_cond_scale,
        )

        if self.temporal_emb:
            video_embed = video_embed.reshape(b, t, d).permute(0, 2, 1)

        text_cond = text if self.decoder_need_text_cond else None
        text_embed = text_embed if self.decoder_need_text_cond else None

        videos = self.decoder.sample(
            video_embed=video_embed,
            text=text_cond,
            text_encodings=text_embed,
            cond_scale=cond_scale,
            cuda=self.decoder_cuda,
        )

        # if one_text:
        #     return first(images)

        return videos
