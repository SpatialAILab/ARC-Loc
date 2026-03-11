import torch
import torch.nn.functional as F
import numpy as np
import math
import os


def create_metric_grid(grid_size, res, batch_size):
    x, y = np.linspace(-grid_size/2, grid_size/2, res), np.linspace(-grid_size/2, grid_size/2, res)
    metric_x, metric_y = np.meshgrid(x, y, indexing='ij')
    metric_x, metric_y = torch.tensor(metric_x).flatten().unsqueeze(0).unsqueeze(-1), torch.tensor(metric_y).flatten().unsqueeze(0).unsqueeze(-1)
    metric_coord = torch.cat((metric_x, metric_y), -1).float()
    return metric_coord.repeat(batch_size, 1, 1)


def desc_l2norm(desc: torch.Tensor) -> torch.Tensor:
    """
    L2-normalize descriptors with shape [N, C] or [N, C, H, W]
    """
    return F.normalize(desc, p=2, dim=1, eps=1e-10)


def save_ckpts(save_path, CVM_model, optimizer, scheduler, current_epoch):
    state = {
        'model': CVM_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedule': scheduler.state_dict(),
        'current_epoch': current_epoch,
    }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(state, save_path)
    print(f"Checkpoint saved to: {save_path}")



