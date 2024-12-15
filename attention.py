import torch
import torch.nn.functional as func
from torch import nn, einsum
from einops import rearrange, repeat


# helper classes
# https://www.53ai.com/news/LargeLanguageModel/2024080656791.html
# PostNorm?
# The class ensures that the input (x) and optionally the context are normalized before being passed to fn.
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(context_dim) if context_dim is not None else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if self.context_norm is not None:
            context = kwargs['context']
            normed_context = self.context_norm(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)

# SwishGLU?
# GLU with GELU as activation
class GEGLU(nn.Module):
    @staticmethod
    def forward(x):
        x, gates = x.chunk(2, dim=-1)
        return x * func.gelu(gates)

# Note: GEGLU() is different from that (i.e., GELU()) in mbt.py
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Preform attention inprint
class Attention(nn.Module):
    #  embed_dim=512, context_dim=512, heads=4, dim_head=512//heads=128
    def __init__(self, embed_dim, context_dim=None, heads=4, dim_head=128, dropout=0.):
        super().__init__()
        # 512 = 4*128
        inner_dim = dim_head * heads
        # 512
        context_dim = context_dim if context_dim is not None else embed_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        # (512, 512)
        self.to_q = nn.Linear(embed_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, embed_dim)

    # x: (batch, feature_dim, channels) = (32, 50, 512)
    # context (wm): (batch, feature_dim, channels) = (32, 50, 512)
    def forward(self, x, context=None, mask=None):
        context = context if context is not None else x
        # q: (batch, feature_dim, channels) * (embed_dim, inner_dim) -> (32, 50, 512) * (512, 512) = (32, 50, 512)
        # # k, v: (32, 50, 512)
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # qkv from (batch, length, (head*head_dim)) to ((batch*head), length, head_dim)
        # qkv: (32, 50, 4*128) -> (32*4, 50, 128)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.heads), (q, k, v))

        # scaled dot-product attention: QK^T
        # (batch*head, length, dk') * (batch*head, dk', length). dk' is the qk vector dim of one attn head
        # (32*4, 50, 128) * (32*4, 128, 50) = (32*4, 50, 50)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # no mask is used
        if mask is not None:
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=self.heads)  # (B*h, 1, T2)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # (32*4, 50, 50) * (32*4, 50, 128) = (32*4, 50, 128)
        out = einsum('b i j, b j d -> b i d', attn, v)
        # attention out: (32, 50, 128*4)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.heads)
        # return: (32, 50, 512)
        return self.to_out(out)

class AttentionImprint(nn.Module):
    def __init__(self, embed_dim, context_dim, depth=2, heads=4, dim_heads=128, ff_expansion=4, attn_dropout=0., ff_dropout=0.):
        super().__init__()
        self.attn_imprint_unit_list = nn.ModuleList([])
        for _ in range(depth):
            self.attn_imprint_unit_list.append(nn.ModuleList([
                PreNorm(dim=embed_dim, fn=Attention(embed_dim, context_dim, heads, dim_heads, attn_dropout), context_dim=context_dim),
                PreNorm(dim=embed_dim, fn=FeedForward(embed_dim, ff_expansion, ff_dropout))
            ]))

    # x: (32, 50, 512)
    # context: (32, 50, 512)
    def forward(self, x, context, context_mask=None):
        for attn, ff in self.attn_imprint_unit_list:
            x = attn(x, context=context, mask=context_mask) + x
            x = ff(x) + x
        # (32, 50, 512)
        return x