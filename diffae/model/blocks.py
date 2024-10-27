import math
from abc import abstractmethod
from dataclasses import dataclass
from numbers import Number

import torch as th
import torch.nn.functional as F
from choices import *
from config_base import BaseConfig
import pdb
import numpy as np

from .nn import (avg_pool_nd, conv_nd, linear, normalization,
                 timestep_embedding, torch_checkpoint, zero_module)


class ScaleAt(Enum):
    after_norm = 'afternorm'


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb=None, cond=None, lateral=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb=emb, cond=cond, lateral=lateral)
            else:
                x = layer(x)
        return x

class projection_layer(nn.Module):
    def __init__(self,
                 in_dim: int = None,
                 out_dim: int = None,
                 is_ortho: bool = False,
                 bias=True, 
                 bias_init=0, 
                 lr_mul=1, 
                 activation=None,
                 diag_size=10):
        super().__init__()
        self.is_ortho = is_ortho
        self.in_dim = in_dim
        self.out_dim = out_dim
    
        if is_ortho:

            self.S = th.zeros(out_dim, in_dim)
            self.S.requires_grad = False
            for i in range(diag_size):
                self.S[i, i] = 1
            self.U = th.nn.Parameter(th.zeros((out_dim, out_dim)).normal_(0, 0.05))
            self.V = th.nn.Parameter(th.zeros((in_dim, in_dim)).normal_(0, 0.05))
            self.scale = 1
        else:
            self.weight = nn.Parameter(th.randn(out_dim, in_dim).div_(lr_mul))
            self.scale = (1 / math.sqrt(in_dim)) * lr_mul

        if bias:
            self.bias = nn.Parameter(th.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.lr_mul = lr_mul
    
    def forward(self, input):

        if self.is_ortho:
            if self.in_dim > self.out_dim:
                weight = self.fasthpp(self.U).mm(self.S.to(input)).mm(self.fasthpp(self.V))
                #weight = Q(self.U).mm(self.S.to(input)).mm(Q(self.V)) #unaccelerated computation
            else:
                weight = self.fasthpp(self.V).mm(self.S.to(input)).mm(self.fasthpp(self.U))
                #weight = Q(self.U).mm(self.S.to(input)).mm(Q(self.V)) #unaccelerated computation

        else:
            weight = self.weight

        input = input.permute(0, 2, 3, 1)
        out = F.linear(
            input, weight * self.scale, bias=self.bias * self.lr_mul
        )
        
        out = out.permute(0, 3, 1, 2)

        return out
    
    def intialize(self, weight_matrix):
        print("weight_matrix shape", weight_matrix.shape)
        if len(weight_matrix.shape) == 4:
            pooled_matrix = F.adaptive_avg_pool2d(weight_matrix, (1, 1))
            weight_matrix = pooled_matrix.squeeze()
        if self.is_ortho:
            with th.no_grad():
                UX, SS, VX = th.svd(weight_matrix, some=False)
                # print("UX shape", UX.shape)
                # print("VX shape", VX.shape)
                # pdb.set_trace()
                hu, tauu = th.geqrf(UX)
                hv, tauv = th.geqrf(VX)
                self.U.data.copy_(hu)
                self.V.data.copy_(hv)
    
    def fasthpp(self, V, stop_recursion=1):
        d = V.shape[0]
        norms = th.norm(V.T, 2, dim=1)
        V = V / norms.view(1, d)
        Y_ = V.clone().T
        W_ = -2 * Y_.clone()

        # Only works for powers of two.
        assert (d & (d - 1)) == 0 and d != 0, "d should be power of two.  "

        # Step 1: compute (Y, W) s by merging!
        k = 1
        for i, c in enumerate(range(int(np.log2(d)))):

            k_2 = k
            k *= 2

            m1_ = Y_.view(d // k_2, k_2, d)[0::2] @ th.transpose(W_.view(d // k_2, k_2, d)[1::2], 1, 2)
            m2_ = th.transpose(W_.view(d // k_2, k_2, d)[0::2], 1, 2) @ m1_

            W_ = W_.view(d // k_2, k_2, d)
            W_[1::2] += th.transpose(m2_, 1, 2)
            W_ = W_.view(d, d)

            if stop_recursion is not None and c == stop_recursion: break
        X = th.eye(d, d, device=V.device)
        # Step 2:
        if stop_recursion is None:
            return X + W_.T @ (Y_ @ X)
        else:
            # For each (W,Y) pair multiply with
            for i in range(d // k - 1, -1, -1):
                X = X + W_[i * k: (i + 1) * k].T @ (Y_[i * k: (i + 1) * k] @ X)
            return X

@dataclass
class ResBlockConfig(BaseConfig):
    channels: int
    emb_channels: int
    dropout: float
    out_channels: int = None
    # condition the resblock with time (and encoder's output)
    use_condition: bool = True
    # whether to use 3x3 conv for skip path when the channels aren't matched
    use_conv: bool = False
    # dimension of conv (always 2 = 2d)
    dims: int = 2
    # gradient checkpoint
    use_checkpoint: bool = False
    up: bool = False
    down: bool = False
    # whether to condition with both time & encoder's output
    two_cond: bool = False
    # number of encoders' output channels
    cond_emb_channels: int = None
    # suggest: False
    has_lateral: bool = False
    lateral_channels: int = None
    # whether to init the convolution with zero weights
    # this is default from BeatGANs and seems to help learning
    use_zero_module: bool = True
    is_ortho: bool = False
    diag_size: int = 10

    def __post_init__(self):
        self.out_channels = self.out_channels or self.channels
        self.cond_emb_channels = self.cond_emb_channels or self.emb_channels

    def make_model(self):
        return ResBlock(self)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    total layers:
        in_layers
        - norm
        - act
        - conv
        out_layers
        - norm
        - (modulation)
        - act
        - conv
    """
    def __init__(self, conf: ResBlockConfig):
        super().__init__()
        self.conf = conf

        #############################
        # IN LAYERS
        #############################
        assert conf.lateral_channels is None
        layers = [
            normalization(conf.channels),
            nn.SiLU(),
            conv_nd(conf.dims, conf.channels, conf.out_channels, 3, padding=1)
        ]

        projection_layers = [
            normalization(conf.channels),
            nn.SiLU(),
            projection_layer(in_dim=512, out_dim=512, bias_init=1, is_ortho=conf.is_ortho, diag_size=conf.diag_size)
        ]

        if conf.is_ortho:
            self.in_layers = nn.Sequential(*projection_layers)
        else:
            self.in_layers = nn.Sequential(*layers)

        self.updown = conf.up or conf.down

        if conf.up:
            self.h_upd = Upsample(conf.channels, False, conf.dims)
            self.x_upd = Upsample(conf.channels, False, conf.dims)
        elif conf.down:
            self.h_upd = Downsample(conf.channels, False, conf.dims)
            self.x_upd = Downsample(conf.channels, False, conf.dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        #############################
        # OUT LAYERS CONDITIONS
        #############################
        if conf.use_condition:
            # condition layers for the out_layers
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                linear(conf.emb_channels, 2 * conf.out_channels),
            )

            if conf.two_cond:
                self.cond_emb_layers = nn.Sequential(
                    nn.SiLU(),
                    linear(conf.cond_emb_channels, conf.out_channels),
                )
            #############################
            # OUT LAYERS (ignored when there is no condition)
            #############################
            # original version
            conv = conv_nd(conf.dims,
                           conf.out_channels,
                           conf.out_channels,
                           3,
                           padding=1)
            if conf.use_zero_module:
                # zere out the weights
                # it seems to help training
                conv = zero_module(conv)

            # construct the layers
            # - norm
            # - (modulation)
            # - act
            # - dropout
            # - conv
            layers = []
            layers += [
                normalization(conf.out_channels),
                nn.SiLU(),
                nn.Dropout(p=conf.dropout),
                conv,
            ]
            self.out_layers = nn.Sequential(*layers)
        
        # self.projector = projection_layer(in_dim=512, out_dim=512, bias_init=1, is_ortho=conf.is_ortho, diag_size=conf.diag_size)

        #############################
        # SKIP LAYERS
        #############################
        if conf.out_channels == conf.channels:
            # cannot be used with gatedconv, also gatedconv is alsways used as the first block
            self.skip_connection = nn.Identity()
        else:
            if conf.use_conv:
                kernel_size = 3
                padding = 1
            else:
                kernel_size = 1
                padding = 0

            self.skip_connection = conv_nd(conf.dims,
                                           conf.channels,
                                           conf.out_channels,
                                           kernel_size,
                                           padding=padding)

    def forward(self, x, emb=None, cond=None, lateral=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Args:
            x: input
            lateral: lateral connection from the encoder
        """
        if x is None:
            print("x is None")
        return torch_checkpoint(self._forward, (x, emb, cond, lateral),
                                self.conf.use_checkpoint)

    def _forward(
        self,
        x,
        emb=None,
        cond=None,
        lateral=None,
    ):
        """
        Args:
            lateral: required if "has_lateral" and non-gated, with gated, it can be supplied optionally    
        """
        if self.conf.has_lateral:
            # lateral may be supplied even if it doesn't require
            # the model will take the lateral only if "has_lateral"
            assert lateral is not None
            x = th.cat([x, lateral], dim=1)

        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)  # [16, 128, 128, 128]

        if self.conf.use_condition:
            # it's possible that the network may not receieve the time emb
            # this happens with autoenc and setting the time_at
            if emb is not None:
                emb_out = self.emb_layers(emb).type(h.dtype)
            else:
                emb_out = None

            if self.conf.two_cond:
                # it's possible that the network is two_cond
                # but it doesn't get the second condition
                # in which case, we ignore the second condition
                # and treat as if the network has one condition
                if cond is None:
                    cond_out = None
                else:
                    # cond shape: [4, 512, 1, 1]
                    try:
                        # cond = cond['cond']
                        cond_out = self.cond_emb_layers(cond).type(h.dtype)
                    except:
                        print("cond:", cond)
                        print("cond shape", cond.shape)
                        print("cond_emb_layers", self.cond_emb_layers)
                        pdb.set_trace()

                if cond_out is not None:
                    while len(cond_out.shape) < len(h.shape):
                        cond_out = cond_out[..., None]
            else:
                cond_out = None

            # this is the new refactored code
            h = apply_conditions(
                h=h,
                emb=emb_out,
                cond=cond_out,
                layers=self.out_layers,
                scale_bias=1,
                in_channels=self.conf.out_channels,
                up_down_layer=None,
            )
        try:
            return self.skip_connection(x) + h
        except:
            print("h dim after projection", h.shape)
            print("res dim", self.skip_connection(x).shape)
            


def apply_conditions(
    h,
    emb=None,
    cond=None,
    layers: nn.Sequential = None,
    scale_bias: float = 1,
    in_channels: int = 512,
    up_down_layer: nn.Module = None,
):
    """
    apply conditions on the feature maps

    Args:
        emb: time conditional (ready to scale + shift)
        cond: encoder's conditional (read to scale + shift)
    """
    two_cond = emb is not None and cond is not None

    if emb is not None:
        # adjusting shapes
        while len(emb.shape) < len(h.shape):
            emb = emb[..., None]

    if two_cond:
        # adjusting shapes
        while len(cond.shape) < len(h.shape):
            cond = cond[..., None]
        # time first
        scale_shifts = [emb, cond]
    else:
        # "cond" is not used with single cond mode
        scale_shifts = [emb]

    # support scale, shift or shift only
    for i, each in enumerate(scale_shifts):
        if each is None:
            # special case: the condition is not provided
            a = None
            b = None
        else:
            if each.shape[1] == in_channels * 2:
                a, b = th.chunk(each, 2, dim=1)
            else:
                a = each
                b = None
        scale_shifts[i] = (a, b)

    # condition scale bias could be a list
    if isinstance(scale_bias, Number):
        biases = [scale_bias] * len(scale_shifts)
    else:
        # a list
        biases = scale_bias

    # default, the scale & shift are applied after the group norm but BEFORE SiLU
    pre_layers, post_layers = layers[0], layers[1:]

    # spilt the post layer to be able to scale up or down before conv
    # post layers will contain only the conv
    mid_layers, post_layers = post_layers[:-2], post_layers[-2:]

    h = pre_layers(h)
    # scale and shift for each condition
    for i, (scale, shift) in enumerate(scale_shifts):
        # if scale is None, it indicates that the condition is not provided
        if scale is not None:
            h = h * (biases[i] + scale)
            if shift is not None:
                h = h + shift
    h = mid_layers(h)

    # upscale or downscale if any just before the last conv
    if up_down_layer is not None:
        h = up_down_layer(h)
    h = post_layers(h)
    return h


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims,
                                self.channels,
                                self.out_channels,
                                3,
                                padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2),
                              mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """
    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims,
                              self.channels,
                              self.out_channels,
                              3,
                              stride=stride,
                              padding=1)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return torch_checkpoint(self._forward, (x, ), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch,
                                                                       dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale,
            k * scale)  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight,
                      v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """
    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5)
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]
