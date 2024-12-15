# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2022 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''ResNet in PyTorch.

Some modifications from the original architecture:
1. Smaller kernel size for the input layer
2. Smaller number of Channels
3. No max_pooling involved

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import pooling_layers as pooling_layers

# https://datawhalechina.github.io/thorough-pytorch/%E7%AC%AC%E5%8D%81%E7%AB%A0/ResNet%E6%BA%90%E7%A0%81%E8%A7%A3%E8%AF%BB.html
# 3x3-->3x3
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1, # if s=1, same padding
                               bias=False)
        # num_features = channel of input = planes
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1, # same padding
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # ensure skip connection path has the same dimension for input and output.
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 1x1-->3x3-->1x1
# input: (in_planes, I1, I2) --> output: (planes*expansion, (I1-1)/stride+1, (I2-1)/stride+1)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# feat_dim: should = num_mel_filter (freq dim of mel spectrogram)
# embed_dim: output dimension, should yield (batch, embed_dim)
class ResNet(nn.Module):
    # Bottleneck, [6, 16, 48, 3], feat_dim = 80, embed_dim = 512, pooling_func = 'MQMHASTP'
    def __init__(self,
                 block,
                 num_blocks,
                 m_channels=32,
                 feat_dim=40,
                 embed_dim=128,
                 pooling_func='TSTP',
                 two_emb_layer=True):
        super(ResNet, self).__init__()
        self.in_planes = m_channels
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        # int(80/8)*32*8 = 2560
        self.stats_dim = int(feat_dim / 8) * m_channels * 8
        self.two_emb_layer = two_emb_layer

        # (32, 1, 80, 50) --> (32, 32, 80, 50)
        self.conv1 = nn.Conv2d(1,
                               m_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1, # same padding
                               bias=False)
        self.bn1 = nn.BatchNorm2d(m_channels)

        # 1st bottleneck block: (32, 32, 80, 50) -> (32, 32*4, 80, 50)
        # 2-6: (32, 128, 80, 50) -> (32, 128, 80, 50) for 5x
        self.layer1 = self._make_layer(block,
                                       m_channels,
                                       num_blocks[0],
                                       stride=1)
        # 1st bottleneck block: (32, 128, 80, 50) -> (32, 32*2*4, 40, 25)
        # 2-16: (32, 256, 40, 25) -> (32, 256, 40, 25) for 15x
        self.layer2 = self._make_layer(block,
                                       m_channels * 2,
                                       num_blocks[1],
                                       stride=2)
        # 1st bottleneck block: (32, 256, 40, 25) -> (32, 32*4*4, 20, 13)
        # 2-48: (32, 512, 20, 13) -> (32, 512, 20, 13) for 47x
        self.layer3 = self._make_layer(block,
                                       m_channels * 4,
                                       num_blocks[2],
                                       stride=2)
        # 1st bottleneck block: (32, 512, 20, 13) -> (32, 32*8*4, 10, 7)
        # 2-3: (32, 1024, 10, 7) -> (32, 1024, 10, 7) for 2x
        self.layer4 = self._make_layer(block,
                                       m_channels * 8,
                                       num_blocks[3],
                                       stride=2)
        # in_dim = 2560*4 =10240
        # (32, 1024, 10, 7) -> (32, 40960)
        self.pool = getattr(pooling_layers,
                            pooling_func)(in_dim=self.stats_dim *
                                          block.expansion)
        # 40960
        self.pool_out_dim = self.pool.get_out_dim()
        # (32, 40960) -> (32, 512)
        self.seg_1 = nn.Linear(self.pool_out_dim, embed_dim)
        if self.two_emb_layer:
            self.seg_bn_1 = nn.BatchNorm1d(embed_dim, affine=False)
            self.seg_2 = nn.Linear(embed_dim, embed_dim)
        else:
            self.seg_bn_1 = nn.Identity()
            self.seg_2 = nn.Identity()

    # Bottleneck, planes=m_channels=32, num_blocks=6,16,48,3, stride=1,2
    def _make_layer(self, block, planes, num_blocks, stride):
        # stride setting applies to the first bottleneck block only.
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            # match the output of previous bottleneck block, 128
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    # x: (32, 50, 80)
    def forward(self, x):
        # (batch_size, num_frames, num_mel_filters) -> (batch_size, num_mel_filters, num_frames)
        # (32, 50, 80) -> (32, 80, 50)
        x = x.permute(0, 2, 1)  # (B,T,F) => (B,F,T)
        # adds one internal channel dimension
        x = x.unsqueeze_(1)
        # (32, 1, 80, 50) --> (32, 32, 80, 50)
        out = F.relu(self.bn1(self.conv1(x)))
        # (32, 32, 80, 50) --> (32, 128, 80, 50)
        out = self.layer1(out)
        # (32, 128, 80, 50) --> (32, 256, 40, 25)
        out = self.layer2(out)
        # (32, 256, 40, 25) --> (32, 512, 20, 13)
        out = self.layer3(out)
        # (32, 512, 20, 13) --> (32, 1024, 10, 7)
        out = self.layer4(out)
        # (32, 1024, 10, 7) -> (32, 40960)
        stats = self.pool(out)
        # (32, 40960) -> (32, 512)
        embed_a = self.seg_1(stats)
        if self.two_emb_layer:
            out = F.relu(embed_a)
            out = self.seg_bn_1(out)
            embed_b = self.seg_2(out)
            # (32, 512), (32, 512)
            return embed_a, embed_b
        else:
            return torch.tensor(0.0), embed_a


def ResNet18(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [2, 2, 2, 2],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet34(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(BasicBlock, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet50(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet101(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet152(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [3, 8, 36, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet221(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [6, 16, 48, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


def ResNet293(feat_dim, embed_dim, pooling_func='TSTP', two_emb_layer=True):
    return ResNet(Bottleneck, [10, 20, 64, 3],
                  feat_dim=feat_dim,
                  embed_dim=embed_dim,
                  pooling_func=pooling_func,
                  two_emb_layer=two_emb_layer)


if __name__ == '__main__':
    x = torch.zeros(10, 200, 80)
    model = ResNet34(feat_dim=80, embed_dim=256, pooling_func='MQMHASTP')
    model.eval()
    out = model(x)
    print(out[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
