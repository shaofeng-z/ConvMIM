import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import sys
sys.path.append("../")
from utils.pos_embed import get_2d_sincos_pos_embed


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self, planes, mlp_ratio=4):
        super(Decoder, self).__init__()
        self.tconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(planes, planes // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes // 2),
            nn.ReLU(inplace=True)
        ) # 14 * 14
        self.tconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(planes // 2, planes // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(inplace=True)
        ) # 28 * 28
        self.tconv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(planes // 4, planes // 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes // 8),
            nn.ReLU(inplace=True)
        ) # 56 * 56
        self.tconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(planes // 8, planes // 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes // 16),
            nn.ReLU(inplace=True)
        ) # 112 * 112
        self.tconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(planes // 16, 3, kernel_size=3, padding=1, bias=True),
        ) # 224 * 224

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, fc_dim=128, in_channel=3, width=1, input_size=224, mix_ratio=0.5, mim=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.base = int(64 * width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.base, layers[0])
        self.layer2 = self._make_layer(block, self.base * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, self.base * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.base * 8, layers[3], stride=2)
        self.out_ftr_size = self.base * 8 * block.expansion
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_ftr_size, fc_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, int(input_size // 4), int(input_size // 4), 64), requires_grad=False)  # fixed sin-cos embedding
        if mim:
            self.mask_embed = nn.Parameter(torch.zeros(1, 64))
        self.mix_ratio = mix_ratio
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.size(1)), cls_token=False).reshape(self.pos_embed.size())
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def interpolate_pos_encoding(self, x):
        b, c, w, h = x.size()
        if w == self.pos_embed.size(1) and h == self.pos_embed.size(2):
            return self.pos_embed
        patch_pos_embed = self.pos_embed
        dim = x.shape[-1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w, h = w + 0.1, h + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.permute(0, 3, 1, 2),
            scale_factor=(w / self.pos_embed.size(1), h / self.pos_embed.size(2)),
            mode='bicubic',
        )
        assert int(w) == patch_pos_embed.shape[-2] and int(h) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1)
        return patch_pos_embed
    
    def add_pos_embed(self, x, position):
        assert x.size(-2) == 56 and x.size(-1) == 56
        b, c, w, h = x.size()
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.size(1)), cls_token=False).reshape(self.pos_embed.size())
        if w == self.pos_embed.size(1) and h == self.pos_embed.size(2):
            return self.pos_embed

    def forward(self, x, target_pos_embed=None):
        """
        target_pos_embed: [b, 64, h ,w]
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x + self.interpolate_pos_encoding(x).permute(0, 3, 1, 2)
        outs = []
        if target_pos_embed is not None:
            masked_query = target_pos_embed + self.mask_embed.unsqueeze(-1).unsqueeze(-1)
            x = self.mix_ratio * masked_query + (1 - self.mix_ratio) * x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet18(fc_dim=128, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], fc_dim = fc_dim , **kwargs)
    return model


def resnet50(fc_dim=128,pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], fc_dim = fc_dim , **kwargs)
    return model

# model = resnet50()
# input_ = torch.randn([4, 3, 224, 224])
# target_pos = torch.randn([4, 64, 56, 56])
# outs = model(input_, target_pos)
# print([out.size() for out in outs])
