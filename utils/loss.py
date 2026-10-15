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


def translation_loss(pred_translation, target_translation):
    """
    Compute the mean Euclidean translation error for each batch item.

    Args:
        pred_translation: Predicted translation with shape [B, 1, 2].
        target_translation: Ground-truth translation with shape [B, 1, 2].

    Returns:
        Tensor of shape [B] containing the translation error.
    """
    displacement = pred_translation - target_translation
    return torch.linalg.vector_norm(displacement, dim=-1).mean(dim=-1)
