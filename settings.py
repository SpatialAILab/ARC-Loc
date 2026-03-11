from typing import Any


import os
import ast
import random

import dataloaders.dataloader_vigor as data_load

from torch.utils.data import DataLoader, random_split
import torch
# import torch.nn as nn
# import torch.optim as optim
from torchvision import transforms
import configparser
from PIL import ImageFile
import numpy as np

import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")

# Handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set deterministic behavior for reproducibility
seed = config.getint("RandomSeed", "seed")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)


GROUND_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "GROUND_IMAGE_SIZE"))
SATELLITE_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "SATELLITE_IMAGE_SIZE"))

res = {'NewYork':0.113248,
        'Seattle':0.100817,
        'SanFrancisco':0.118141,
        'Chicago':0.111262,
        }

# Define transformations
transform_grd = transforms.Compose([
    transforms.Resize(GROUND_IMAGE_SIZE),
    transforms.ToTensor()
])

transform_sat = transforms.Compose([
    transforms.Resize(SATELLITE_IMAGE_SIZE),
    transforms.ToTensor()
])

NewYork_res = 0.113248 
Seattle_res = 0.100817
SanFrancisco_res = 0.118141 
Chicago_res = 0.111262 




def fetch_dataloader_VIGOR(args, train_dataset, split, is_point=False):

    print('Training with %d image pairs' % len(train_dataset))

    if split == 'train':
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print("using {} images for training, {} images for validation.".format(train_size, val_size))
        nw = 12
        if is_point == True:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw, collate_fn=data_load.sparse_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw, collate_fn=data_load.sparse_collate_fn)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

        return train_loader, val_loader
    
    else: # test time
        nw = 12  # number of workers
        print('Using {} dataloader workers every process'.format(nw)) 
        if is_point == True:
            test_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                pin_memory=True, shuffle=False, num_workers=nw, drop_last=False, collate_fn=data_load.sparse_collate_fn)    
        else:
            test_loader = DataLoader[Any](train_dataset, batch_size=args.batch_size,
                        pin_memory=True, shuffle=False, num_workers=nw, drop_last=False)    

        return test_loader
    


def fetch_optimizer(args, CVM_model):

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(
                                CVM_model.parameters(), 
                                lr=args.lr, 
                                weight_decay=0.01
                                )
        print('AdamW Optimizer')
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(
                                    CVM_model.parameters(), 
                                    lr=args.lr, 
                                    betas=(0.9, 0.999),
                                    )
        print('Adam Optimizer')
        print('lr =',args.lr)

    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch)
        print('Cosine Optimizer')

    return optimizer, scheduler


def grid_size(dataset):
    if dataset=='VIGOR':
        h, v = 71, 20
    
    elif dataset=='kitti':
        h, v = 100, 20

    return h, v


def restore_ckpts(args, CVM_model, optimizer, scheduler):
    checkpoint = torch.load(args.restore_ckpt, map_location='cpu')
    # best_dis = checkpoint['best_dis']
    current_epoch = checkpoint['current_epoch']

    CVM_model.load_state_dict(checkpoint['model'])
    CVM_model = CVM_model.to(f'cuda:{args.gpuid}')

    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_schedule'])
    
    print("Have load state_dict from: {}".format(args.restore_ckpt))

    return CVM_model, optimizer, scheduler, current_epoch


# ================================== SETTINGS ==================================

def set_loader(args):
    '''
    Load elements for training/evaluation
    optim : AdamW or Adam(default) / cos, reduce, oncycle, multicycle scheduler (default settings no use scheduler)
    data : VIGOR, VIGOR-aerial (different return image size)
    '''
    args.scheduler = 'cos'

    if args.settings == 'ARC_Loc_baseline': # ---------------------------------------------------------

        from models.ARC2.ARC_Loc import CVM
        from hubconf import radio_model


        #### 1. Model
        try: # linux
            extractor = torch.compile(radio_model(version='c-radio_v3-l').half())
            # extractor = torch.compile(radio_model(version='c-radio_v3-l'))
            print('torch.compile for RADIO')
        except Exception: # window
            extractor = radio_model(version='c-radio_v3-l')
            
        # extractor = radio_model(version='c-radio_v3-l')
        # extractor = radio_model(version='c-radio_v3-l').half()
        
        data_type = 'VIGOR'

        grid_size_h, grid_size_v = grid_size(data_type)

        model = CVM(args.gpuid, 
                    grd_enc=extractor,
                    sat_enc=extractor,
                    grd_bev_res=args.grd_bev_res, 
                    sat_bev_res=args.sat_bev_res, 
                    grid_size_h=grid_size_h, 
                    grid_size_v=grid_size_v,
                    )

        # 2. Pre-trained weights and Optimizer
        optimizer, scheduler = fetch_optimizer(args, model)

        if args.restore_ckpt is not None:
            print('restoreing...')
            model, optimizer, scheduler, current_epoch = restore_ckpts(args, model, optimizer, scheduler)
        else: 
            current_epoch = 0


        # 3. Dataset
        train_dataset = data_load.VIGORArcDataset(root=args.dataset_root, label_root=args.label_root, split=args.area, train=args.is_train, \
                            transform=(transform_grd, transform_sat), random_orientation=args.random_orientation, use_filtered=args.use_filtered)
        if args.is_train =='train':
            train_loader, val_loader = fetch_dataloader_VIGOR(args, train_dataset, split=args.is_train)
            return model, train_loader, val_loader, optimizer, scheduler, data_type, current_epoch
        
        else: # test time
            test_loader = fetch_dataloader_VIGOR(args, train_dataset, split=args.is_train)
            return model, test_loader 


    elif args.settings == 'ARC_Loc_kitti': # ---------------------------------------------------------

        from models.ARC2.ARC_Loc import CVM
        from hubconf import radio_model

        data_type = 'kitti'

        # 1. Model
        try: # linux
            extractor = torch.compile(radio_model(version='c-radio_v3-l'))
        except Exception: # window
            extractor = radio_model(version='c-radio_v3-l')

        grid_size_h, grid_size_v = grid_size('kitti')

        model = CVM(args.gpuid, 
                    grd_enc=extractor,
                    sat_enc=extractor,
                    grd_bev_res=args.grd_bev_res, 
                    sat_bev_res=args.sat_bev_res, 
                    grid_size_h=grid_size_h, 
                    grid_size_v=grid_size_v,
                    )

        # 2. Pre-trained weights and Optimizer
        optimizer, scheduler = fetch_optimizer(args, model)

        if args.restore_ckpt is not None:
            print('restoreing...')
            model, optimizer, scheduler, current_epoch = restore_ckpts(args, model, optimizer, scheduler)
        else: 
            current_epoch = 0


        # 3. Dataset
        train_dataset = 0
        if args.is_train =='train':
            return model, 0, 0, optimizer, scheduler, data_type, current_epoch
        
        else: # test time
            return model, 0 

    elif args.settings == 'ARC_Loc_attn_kitti': # ---------------------------------------------------------

        from models.ARC2.ARC_Loc_attn import CVM
        from hubconf import radio_model

        data_type = 'kitti'

        # 1. Model
        try: # linux
            extractor = torch.compile(radio_model(version='c-radio_v3-l'))
        except Exception: # window
            extractor = radio_model(version='c-radio_v3-l')
        

        grid_size_h, grid_size_v = grid_size('kitti')

        model = CVM(args.gpuid, 
                    grd_enc=extractor,
                    sat_enc=extractor,
                    grd_bev_res=args.grd_bev_res, 
                    sat_bev_res=args.sat_bev_res, 
                    grid_size_h=grid_size_h, 
                    grid_size_v=grid_size_v,
                    )

        # 2. Pre-trained weights and Optimizer
        optimizer, scheduler = fetch_optimizer(args, model)

        if args.restore_ckpt is not None:
            print('restoreing...')
            model, optimizer, scheduler, current_epoch = restore_ckpts(args, model, optimizer, scheduler)
        else: 
            current_epoch = 0


        # 3. Dataset
        train_dataset = 0
        if args.is_train =='train':
            return model, 0, 0, optimizer, scheduler, data_type, current_epoch
        
        else: # test time
            return model, 0 




    elif args.settings == 'ARC_Loc_attn': # ---------------------------------------------------------

        from models.ARC2.ARC_Loc_attn import CVM
        from hubconf import radio_model

        data_type = 'VIGOR'


        # 1. Model
        try: # linux
            extractor = torch.compile(radio_model(version='c-radio_v3-l'))
        except Exception: # window
            extractor = radio_model(version='c-radio_v3-l')
        

        grid_size_h, grid_size_v = grid_size(data_type)

        model = CVM(args.gpuid, 
                    grd_enc=extractor,
                    sat_enc=extractor,
                    grd_bev_res=args.grd_bev_res, 
                    sat_bev_res=args.sat_bev_res, 
                    grid_size_h=grid_size_h, 
                    grid_size_v=grid_size_v,
                    )

        # 2. Pre-trained weights and Optimizer
        optimizer, scheduler = fetch_optimizer(args, model)

        if args.restore_ckpt is not None:
            print('restoreing...')
            model, optimizer, scheduler, current_epoch = restore_ckpts(args, model, optimizer, scheduler)
        else: 
            current_epoch = 0


        # 3. Dataset
        train_dataset = data_load.VIGORArcDataset(root=args.dataset_root, label_root=args.label_root, split=args.area, train=args.is_train, \
                            transform=(transform_grd, transform_sat), random_orientation=args.random_orientation, use_filtered=args.use_filtered)
        if args.is_train =='train':
            train_loader, val_loader = fetch_dataloader_VIGOR(args, train_dataset, split=args.is_train)
            return model, train_loader, val_loader, optimizer, scheduler, data_type, current_epoch
        
        else: # test time
            test_loader = fetch_dataloader_VIGOR(args, train_dataset, split=args.is_train)
            return model, test_loader 


