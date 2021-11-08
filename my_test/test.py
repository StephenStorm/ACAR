import torch
import torch.nn as nn
feats = [torch.zeros((1, 2048, 2, 29, 16)), torch.ones((1, 256, 5, 29, 16))] 
h, w = feats[0].shape[3:]
feats = [nn.AdaptiveAvgPool3d((1, h, w))(f).view(-1, f.shape[1], h, w) for f in feats]
print(feats[0].shape, feats[1].shape)
feats = torch.cat(feats, dim=1)
print(feats.shape)