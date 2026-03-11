import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import desc_l2norm
from torch.hub import load_state_dict_from_url
from att_layers.transformer import Transformer_self_att

import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")
import ast

ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))
eps = config.getfloat("Constants", "epsilon")


seed = config.getint("RandomSeed", "seed")
import random
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)



class BasicBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bn=True, padding_mode='zeros'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm2d(planes) if bn else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(planes) if bn else nn.Identity()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, padding_mode=padding_mode)
            )

    def forward(self, x, relu=True):
        shortcut = self.shortcut(x) if hasattr(self, 'shortcut') else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += shortcut
        if relu:
            out = F.relu(out)
        return out


class DeepResBlock_desc(torch.nn.Module):
    def __init__(self, bn, last_dim, in_channels, block_dims, add_posEnc, norm_desc, padding_mode = 'zeros'):
        super().__init__()

        self.norm_desc = norm_desc

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], last_dim, stride=1, bn=bn, padding_mode=padding_mode)


        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)


    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x, relu=False)

        if self.norm_desc:
            x = desc_l2norm(x)

        return x



class DeepResBlock_kpts(torch.nn.Module):
    def __init__(self, bn, in_channels, block_dims, add_posEnc, use_softmax, padding_mode='zeros'):
        super().__init__()

        self.resblock1 = BasicBlock(in_channels, block_dims[0], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], stride=1, bn=bn, padding_mode=padding_mode)
        self.resblock4 = BasicBlock(block_dims[2], block_dims[3], stride=1, bn=bn, padding_mode=padding_mode)

        self.score = nn.Conv2d(block_dims[3], 1, kernel_size=1, stride=1, padding=0, bias=False)

        self.use_softmax = use_softmax
        self.sigmoid = torch.nn.Sigmoid()
        self.logsigmoid = torch.nn.LogSigmoid()
        self.softmax = torch.nn.Softmax(dim=-1)

        # Allow more exploration with reinforce algorithm
        self.tmp_softmax = 10

        self.eps = nn.Parameter(torch.tensor(1e-16), requires_grad=False)
        self.offset_par1 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.offset_par2 = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.ones_kernel = nn.Parameter(torch.ones((1, 1, 3, 3)), requires_grad=False)

        self.att_layer = Transformer_self_att(d_model=128, num_layers=3, add_posEnc=add_posEnc)

    def remove_borders(self, score_map: torch.Tensor, borders: int):
        '''
        It removes the borders of the image to avoid detections on the corners
        '''
        shape = score_map.shape
        mask = torch.ones_like(score_map)

        mask[:, :, 0:borders, :] = 0
        mask[:, :, :, 0:borders] = 0
        mask[:, :, shape[2] - borders:shape[2], :] = 0
        mask[:, :, :, shape[3] - borders:shape[3]] = 0

        return mask * score_map

    def remove_brd_and_softmax(self, scores, borders):

        B = scores.shape[0]

        scores = scores - (scores.view(B, -1).mean(-1).view(B, 1, 1, 1) + self.eps).detach()
        exp_scores = torch.exp(scores / self.tmp_softmax)

        # remove borders
        exp_scores = self.remove_borders(exp_scores, borders=borders)

        # apply softmax
        sum_scores = exp_scores.sum(-1).sum(-1).view(B, 1, 1, 1)
        return exp_scores / (sum_scores + self.eps)

    def forward(self, feature_volume):

        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)
        x = self.resblock4(x)

        # Predict xy scores
        scores = self.score(x)

        if self.use_softmax:
            scores = self.remove_brd_and_softmax(scores, 3)
        else:
            scores = self.remove_borders(self.sigmoid(scores), borders=3)

        return scores


class Upsample1p5(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        H2 = int(round(H * 1.5))
        W2 = int(round(W * 1.5))
        return F.interpolate(x, size=(H2, W2),
                            mode='bilinear',
                            align_corners=False)


class DeepResBlock_desc_Up(nn.Module):
    def __init__(self, bn, last_dim, in_channels, block_dims,
                add_posEnc, norm_desc, padding_mode='zeros'):
        super().__init__()

        self.norm_desc = norm_desc

        self.resblock1 = BasicBlock(in_channels,   block_dims[0], 1, bn, padding_mode=padding_mode)
        self.resblock2 = BasicBlock(block_dims[0], block_dims[1], 1, bn, padding_mode=padding_mode)
        self.resblock3 = BasicBlock(block_dims[1], block_dims[2], 1, bn, padding_mode=padding_mode)

        self.att_layer = Transformer_self_att(
            d_model=block_dims[2],  
            num_layers=3,
            add_posEnc=add_posEnc
        )

        self.upsample_mid = Upsample1p5()

        # 마지막 블록은 고해상도에서 descriptor를 생성
        self.resblock4 = BasicBlock(block_dims[2], last_dim, 1, bn, padding_mode=padding_mode)

    def forward(self, feature_volume):
        x = self.resblock1(feature_volume)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.att_layer(x)

        x = self.upsample_mid(x)

        x = self.resblock4(x, relu=False)

        if self.norm_desc:
            x = desc_l2norm(x)

        return x
