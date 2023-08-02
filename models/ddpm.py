import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
import numpy as np

import torch
from torch import nn, einsum
import torch.nn.functional as F

# taken and adapted from https://huggingface.co/blog/annotated-diffusion

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = (
            time * embeddings[None, :]
        )  # time is already in the shape [batchsize,1]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianFourierProjection(nn.Module): # for continuous training
  """Gaussian Fourier embeddings for noise levels.
     taken from https://github.com/yang-song/score_sde_pytorch/blob/cb1f359f4aadf0ff9a5e122fe8fffc9451fd6e44/models/layerspp.py#L32
  """

  def __init__(self, dim=256, scale=16.0):
    super().__init__()
    dim = dim//2
    self.W = nn.Parameter(torch.randn(dim) * scale, requires_grad=False)

  def forward(self, x):
    x_proj = x* self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(
        self,
        dim,
        dim_out,
        *,
        time_emb_dim=None,
        groups=8,
        class_label_cond=False,
        num_classes=None
    ):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out)) if exists(time_emb_dim) else None

        if class_label_cond:
            # TODO time_emb_dim, class_emd_dim should have their own params for dimentionality.
            self.class_label_mlp = nn.Sequential(
                nn.SiLU(), nn.Linear(time_emb_dim, dim_out)
            )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

        self.num_classees = num_classes
        self.class_label_cond = class_label_cond

    def forward(self, x, time_emb=None, class_lbl=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        if self.class_label_cond is True:
            class_lbl_emb = self.class_label_mlp(
                class_lbl
            )  # Bring the lbl_emb to correct shape.
            h = rearrange(class_lbl_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Network(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1, # channels of sdf input
        cond_channels=1, # rgb vs gray input for conditioning
        embedding='sinusoidal',
        with_time_emb=True,
        resnet_block_groups=2,
        img_cond=None,
        with_class_label_emb=False,
        class_label_cond=False,
        num_classes=None,
    ):
        super().__init__()

        self.embedding = embedding
        assert self.embedding == 'fourier' or self.embedding == 'sinusoidal'

        self.class_label_cond = class_label_cond
        self.num_classes = num_classes
        self.with_class_label_emb = with_class_label_emb

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                GaussianFourierProjection(dim) if self.embedding == 'fourier' else SinusoidalPositionEmbeddings(dim), # TODO: include option which embedding type to use
                # SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            raise Exception("Time embedding is set to False, None of the other code can deal with it. Think. Idiot.")

        # class_label embeddings
        if class_label_cond == True:
            if with_class_label_emb:
                class_label_dim = dim * 4
                self.class_label_embedding_mlp = nn.Sequential(
                    SinusoidalPositionEmbeddings(dim),
                    nn.Linear(dim, class_label_dim),
                    nn.GELU(),
                    nn.Linear(class_label_dim, class_label_dim),
                )
            else:
                class_label_dim = dim * 4
                self.class_label_embedding_mlp = nn.Sequential(
                    nn.Linear(dim, class_label_dim),
                    nn.GELU(),
                    nn.Linear(class_label_dim, class_label_dim),
                )
        else:
            self.class_label_embedding_mlp = None

        # determine dimensions
        self.channels = channels
        self.cond_channels = cond_channels

        # conditioning branch at beginning (SegDiff style)
        if img_cond == 1:
            self.encoder_img = nn.Sequential(
                block_klass(channels, init_dim, time_emb_dim=time_dim)
            )
            self.encoder_mask = nn.Sequential(
                block_klass(cond_channels, init_dim, time_emb_dim=time_dim)
            )
            init_dim *= 2
            channels = init_dim

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_in,
                            dim_out,
                            time_emb_dim=time_dim,
                            class_label_cond=class_label_cond,
                            num_classes=num_classes,
                        ),
                        block_klass(
                            dim_out,
                            dim_out,
                            time_emb_dim=time_dim,
                            class_label_cond=class_label_cond,
                            num_classes=num_classes,
                        ),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(
                            dim_out * 2,
                            dim_in,
                            time_emb_dim=time_dim,
                            class_label_cond=class_label_cond,
                            num_classes=num_classes,
                        ),
                        block_klass(
                            dim_in,
                            dim_in,
                            time_emb_dim=time_dim,
                            class_label_cond=class_label_cond,
                            num_classes=num_classes,
                        ),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, self.channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim, time_emb_dim=None), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, img_cond=None, class_lbl=None):
        if img_cond is not None:
            # x = self.encoder_img(x) + self.encoder_mask(cond)
            x = torch.cat((self.encoder_img(x), self.encoder_mask(img_cond)), 1)

        x = self.init_conv(x)

        t = self.time_mlp(time) if exists(self.time_mlp) else None
        class_lbl = (
            self.class_label_embedding_mlp(class_lbl)
            if exists(self.class_label_embedding_mlp)
            else None
        )

        h = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, class_lbl)
            x = block2(x, t, class_lbl)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t, class_lbl)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, class_lbl)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, class_lbl)
            x = block2(x, t, class_lbl)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)