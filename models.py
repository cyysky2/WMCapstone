import torch
import torch.nn.functional as func
import torch.nn as nn
from utils import init_weights, get_padding
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from attention import AttentionImprint

LRELU_SLOPE = 0.1

class ResBlock1(torch.nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.cfg = cfg
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)


    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = func.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = func.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

class ResBlock2(torch.nn.Module):
    def __init__(self, cfg, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.cfg = cfg
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = func.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Encoder(nn.Module):
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.num_kernels = len(self.cfg.resblock_kernel_sizes)
        self.num_upsamples = len(self.cfg.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(1, 32, 7, 1, padding=3))
        self.normalize = nn.ModuleList()

        res_block = ResBlock1 if self.cfg.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(list(reversed(list(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes))))):
            # double the channel num, keeps temporal dimension
            self.ups.append(weight_norm(Conv1d(32*(2**i), 32*(2**(i+1)), k, u, padding = (k-u)//2)))

        self.res_blocks = nn.ModuleList()
        for i in range(len(self.ups)):
            res_in_ch = 32*(2**(i+1))
            for j , (k, d) in enumerate(zip(
                    list(reversed(cfg.resblock_kernel_sizes)),
                    list(reversed(cfg.resblock_dilation_sizes)))):
                self.res_blocks.append(res_block(cfg, res_in_ch, k, d)) # in_ch = out_ch
                self.normalize.append(nn.GroupNorm(res_in_ch // 16, res_in_ch, eps=1e-6, affine=True))

        self.conv_post = Conv1d(512, 512, 3, 1, padding=1)
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.AIU = AttentionImprint(embed_dim=512, context_dim=512,
                                    depth=2, heads=4, dim_heads=512//4,
                                    ff_expansion=4, attn_dropout=0., ff_dropout=0.)

    # watermark: (32, 50, 512)
    def forward(self, x, watermark):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = func.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xt = None
            for j in range(self.num_kernels):
                if xt is None:
                    xt = self.res_blocks[i * self.num_kernels + j](x)
                    xt = self.normalize[i * self.num_kernels + j](xt)
                else:
                    xt += self.res_blocks[i * self.num_kernels + j](x)
                    xt = self.normalize[i * self.num_kernels + j](xt)
            x = xt / self.num_kernels
        x = func.leaky_relu(x)
        x = self.conv_post(x) # [32, 512, 50]

        # Attention imprint
        # # [32, 512] ——> [32, 1, 512]
        # watermark = watermark.unsqueeze(1)
        # # TODO: take out this repeat operation
        # # [32, 1, 512] ——> [32, 50, 512]
        # watermark = watermark.expand(watermark.shape[0], x.shape[2], watermark.shape[2])

        # Attention imprint
        # [32, 512, 50] ——> [32, 50, 512]
        x = x.transpose(1, 2)
        # [32, 50, 512] and [32, 50, 512] to [32, 50, 512]
        x = self.AIU(x, watermark)
        # [32, 50, 512] -> [32, 512, 50]
        x = x.transpose(1, 2)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)


class QuantizerModule(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super().__init__()
        # n_e: number of embeddings (size of codebook),
        # e_dim: embedding dimensions (embed vec dim)
        # (1024, 512/1)
        self.embedding = nn.Embedding(n_e, e_dim)
        # Embedding weights are initialized uniformly in the range [-1.0 / n_e, 1.0 / n_e]
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)  # [1024, 512]

    # x: [1600, 512]
    def forward(self, x):
        # compute Euclidean distance
        # (1600, 1024)
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indexes = torch.argmin(d, 1)  # [1600]
        # sample 1600 approximations from the embedding table.
        z_q = self.embedding(min_indexes) # [1600, 512/1]
        # Training: (1600, 512/1), (1600); Validation: (32*T, 512/1), (32*T)
        return z_q, min_indexes


class Quantizer(torch.nn.Module):
    def __init__(self, h):
        super(Quantizer, self).__init__()

        self.h = h
        self.n_code_groups = self.h.Audio["n_code_groups"]
        self.n_codes = self.h.Audio["n_codes"]
        self.codebook_loss_lambda = self.h.Audio["codebook_loss_lambda"]
        self.commitment_loss_lambda = self.h.Audio["commitment_loss_lambda"]
        self.codebook_weight = self.h.Audio["codebook_weight"]
        self.residual_layer = int(self.h.Audio["residual_layer"])

        assert self.codebook_weight % self.n_code_groups == 0

        self.quantizer_module_residual_list = nn.ModuleList()
        for i in range(self.residual_layer):
            self.quantizer_module_residual_list.append(nn.ModuleList([
                QuantizerModule(self.n_codes, self.codebook_weight // self.n_code_groups) for _ in
                range(self.n_code_groups)
            ]))

    def for_one_step(self, xin, idx):
        # training: (32, 50, 512), validation: (32, T, 512)
        xin = xin.transpose(1, 2)
        # (32*50, 512) = (1600, 512)
        x = xin.reshape(-1, self.codebook_weight)
        # (1600, 512/1)
        x = torch.split(x, self.codebook_weight // self.n_code_groups, dim=-1)
        min_indexes_list = []
        z_q = []

        for _x, m in zip(x, self.quantizer_module_residual_list[idx]):
            # training: [1600, 512/1], [1600]; Validation: (32*T, 512/1), (32*T)
            _z_q, min_indexes = m(_x)
            z_q.append(_z_q)
            min_indexes_list.append(min_indexes)  # B * T,
        # n_group * (32*T, 512/n_group) -> (32*T, 512) -> (32, T, 512)
        z_q = torch.cat(z_q, -1).reshape(xin.shape)
        # loss = 0.25 * torch.mean((z_q.detach() - xin) ** 2) + torch.mean((z_q - xin.detach()) ** 2)
        loss = self.codebook_loss_lambda * torch.mean((z_q - xin.detach()) ** 2) \
               + self.commitment_loss_lambda * torch.mean((z_q.detach() - xin) ** 2)
        z_q = xin + (z_q - xin).detach()
        z_q = z_q.transpose(1, 2)
        # (32, T, 512), 1, group*32*T = 32*T
        return z_q, loss, min_indexes_list

    # 1st xin: [32, 512, 50]
    def forward(self, xin):
        # B, C, T
        quantized_out = 0.0
        residual = xin
        all_losses = []
        all_indices = []
        for i in range(self.residual_layer):
            quantized, loss, indices = self.for_one_step(residual, i)  #
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.extend(indices)  # extend(32*T*group)
            all_losses.append(loss)
        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)
        # (32, T, 512), 1, (#residual_layer, #group, 32, T) = (16, 1, 32, 50)
        return quantized_out, loss, all_indices


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # starting from upsample_initial_channel=512
        self.conv_pre = weight_norm(
            Conv1d(h.Audio["codebook_weight"], h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(
                    h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                    k, u,
                    padding=(k - u) // 2,
                )
            ))

        self.resblocks = nn.ModuleList()
        ch = 0
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = func.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xt = None
            for j in range(self.num_kernels):
                if xt is None:
                    xt = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xt += self.resblocks[i * self.num_kernels + j](x)
            x = xt / self.num_kernels
        x = func.leaky_relu(x, LRELU_SLOPE)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


# the generator is encouraged to replicate the hierarchical feature maps of real audio
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

# minimize the difference between the discriminator's generated and the real labels.
# disc_outputs: the label output of discriminator from audio generated by the generator.
# i.e., make the generator thinks that the audio generated by the generator is all real.
def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses