import torch

import os


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



