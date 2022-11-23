# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import sys
sys.path.append("../")
from utils.resnet import resnet18, resnet50, Decoder

class ConvMIM(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, contra_dim=2048, pred_dim=512, decoder_dim=512, mim=True, mix=0.5):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(ConvMIM, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = resnet50(fc_dim=contra_dim, mix_ratio=mix)
        self.mim = mim

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(contra_dim, affine=False)) # output layer
        self.projector.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(contra_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, contra_dim)) # output layer

        # if self.mim:
        #     self.decoder = Decoder(decoder_dim)

    def forward_contrastive_mim_loss(self, anchor_map, target_map):
        """
        Input: B, C, H, W
        """
        criterion = nn.CosineSimilarity(dim=-1).cuda(anchor_map.device)
        z1 = anchor_map.permute(0, 2, 3, 1)
        z2 = target_map.permute(0, 2, 3, 1)
        z1 = self.projector(z1)
        z2 = self.projector(z2)
        p1 = self.predictor(z1) # NxC
        return 1 - criterion(p1, z2.detach()).mean()

    def forward(self, anchor_imgs, target_imgs, target_pos):
        """
        Input:
            anchor_imgs: first views of images
            target_imgs: second views of images
            target_pos: relative position of target_imgs to anchor_imgs
        Output:
            contrastive_loss and reconstruct_loss
        """

        # compute features for one view
        anchor_map = self.encoder(anchor_imgs, target_pos) # NxC
        target_map = self.encoder(target_imgs) # NxC
        return self.forward_contrastive_mim_loss(anchor_map, target_map)
        
