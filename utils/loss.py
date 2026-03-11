import torch
import torch.nn.functional as F
import configparser
import torch.nn as nn
from utils.utils import *
config = configparser.ConfigParser()
config.read("./config.ini")

# grd_bev_res = config.getint("VIGOR", "grd_bev_res")
sat_bev_res = config.getint("VIGOR", "sat_bev_res")
grid_size_h = config.getfloat("VIGOR", "grid_size_h")
eps = config.getfloat("Constants", "epsilon")


def compute_vce_loss(X0, Rgt, tgt, R, t):
    """
    Computes Virtual Correspondence Error loss between ground-truth and predicted transformations.

    Args:
        X0 (Tensor): Initial 3D coordinates [B, N, 3].
        Rgt (Tensor): Ground-truth rotation matrix [B, 3, 3].
        tgt (Tensor): Ground-truth translation vector [B, 1, 3].
        R (Tensor): Predicted rotation matrix [B, 3, 3].
        t (Tensor): Predicted translation vector [B, 1, 3].

    Returns:
        loss (Tensor): Mean reprojection error per batch.
    """
    B = X0.shape[0]

    # Transform points using ground-truth and predicted transformations
    X1_gt = Rgt @ X0.repeat(B,1,1).transpose(2, 1) + tgt.transpose(2, 1) # tgt:2,1,2 | Rgt@X0:2,2,1681
    X1_pred = R @ X0.repeat(B,1,1).transpose(2, 1) + t.transpose(2, 1) 

    # Compute L2 distance 
    loss = torch.mean(torch.sqrt(((X1_gt - X1_pred)**2).sum(dim=1)), dim=-1)

    return loss
