import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import numpy as np
import math
import random

import torch
import torch.nn as nn

from utils.loss import *
from utils.utils import *
from utils.solver import *

from tqdm import tqdm
import time

from settings import set_loader
from evaluate_func import validation_Arc as validation

def train(args, gsd):
    device = f'cuda:{args.gpuid}'
    num_samples_matches = args.num_samples 

    # --- 0. Setup ---
    CVM_model, train_loader, valid_loader, optimizer, scheduler, data_type, current_epoch_ = set_loader(args)

    if args.restore_ckpt is not None:
        print(f'========== Load CKPT : {args.restore_ckpt} ==========')
        current_epoch = current_epoch_
    else:
        current_epoch = 0

    CVM_model.to(device)

    results_dir = os.path.join('results', 'vigor', args.area, 'known_ori')
    os.makedirs(results_dir, exist_ok=True)

    # 0. define a metric grid 
    metric_coord4loss = create_metric_grid(grid_size_h, grd_bev_res[0], 1).to(device)


    best_dist_mean = 10000
    best_dist_median = 10000
    for epoch in range(current_epoch, args.num_epoch):
        CVM_model.train()

        running_loss = 0.0
        running_ray_dist = 0.0
        
        num_batches = 0

        pbar = tqdm(train_loader, dynamic_ncols=True)


        for it, data in enumerate(pbar):
            grd, sat, tgt, Rgt, city, etc = data 
            etc.append(epoch) 

            grd = grd.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)
            

            B, _, sat_size, _ = sat.size()
            center = sat_size / 2


            tgt_px = tgt.clone() 
            tgt_px[:,:,0] = center - tgt[:,:,0] # y(h)
            tgt_px[:,:,1] = center - tgt[:,:,1] # x(w)


            # --- 1. Feature matching ---
            matching_score, matching_score_original, etc_output = CVM_model(grd, sat, etc)
            B, num_kpts_sat, num_kpts_grd = matching_score_original.shape

            # --- 2. top-N feature sampling ---
            matches_flat = matching_score.flatten(1)
            batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
            sampled_idx = torch.multinomial(matches_flat, num_samples_matches)

            sampled_sat_idx = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_grd_idx = sampled_idx % num_kpts_grd

            grd_feat_x_size = 2*int((num_kpts_grd//2)**(1/2)) # feature w size

            sampled_grd_x_idx = sampled_grd_idx % grd_feat_x_size

            weights = matches_flat[batch_idx, sampled_idx]

            # --- 3. Extract azimuth of selected grd desc's ---
            u_g = pano_dirs_from_indices(sampled_grd_x_idx, grd_feat_x_size) # b, k, 2, dir(x,y)

            if args.random_orientation:
                u_g = u_g @ Rgt # row vector rotation : CW
            
            # --- 4. Pose estimation ---
            pstars, dist_loss = ARC_Solver(
                sampled_sat_idx=sampled_sat_idx,   # (B,K)
                u_ground=u_g,                 # r(B,K,2)
                Hs=sat_size,
                sat_bev_res=sat_bev_res,
                weights=weights,               # (B,K) 또는 None
            )







            # --- 4. loss function & gradient ---

            trans = pstars.clone() # pstars : pixel level coord
            # pixel to translation
            trans[:,0] = center - pstars[:,1] 
            trans[:,1] = center - pstars[:,0]
            trans = trans.unsqueeze(1) # trans : meter level translation
            t = trans/sat_size * grid_size_h # pixel level translation >> meter level translation?


            losses = []

            if args.vce_loss :
                loss_vce = compute_vce_loss(metric_coord4loss, Rgt, t, Rgt, tgt*grid_size_h/sat_size)
                losses.append(loss_vce.mean())

            if args.ray_loss: 

                loss_ray_dist = dist_loss.mean()
                losses.append(args.alpha * loss_ray_dist)

            loss = sum(losses)
            
            num_batches += 1
            running_loss += loss.item()
            avg_loss = running_loss / num_batches

            if args.ray_loss:
                running_ray_dist += loss_ray_dist.item()
                avg_ray_dist = running_ray_dist / num_batches

            pbar.set_postfix({
                'total_loss': avg_loss,
                'dist_loss': round(avg_ray_dist, 3) if args.ray_loss else '-',
            })


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()





        # ============= Validation & Save point =============
        print(f'========== {epoch} Validation ==========')
        t_error, t_median_error = validation(args, CVM_model, valid_loader, results_dir, device)
    
        if t_error < best_dist_mean:
            best_dist_mean = t_error

            save_path = f'checkpoints/{args.area}/{ori_name}/' + f'Best_mean_ckpt_{args.settings}_{args.name}.pth'
            save_ckpts(
                save_path = save_path,
                CVM_model=CVM_model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_epoch=epoch,
                )
        if t_median_error < best_dist_median:
            best_dist_median = t_median_error

            save_path = f'checkpoints/{args.area}/{ori_name}/' + f'Best_median_ckpt_{args.settings}_{args.name}.pth'
            save_ckpts(
                save_path = save_path,
                CVM_model=CVM_model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_epoch=epoch,
                )
        if args.scheduler:
            scheduler.step()

    return




if __name__ == '__main__':
    import configparser
    import ast

    seed = 777
    print('seed number:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Load configuration

    config = configparser.ConfigParser()
    config.read("./config.ini")

    dataset_root = config["VIGOR"]["dataset_root"]
    label_root = config["VIGOR"]["label_root"]

    grd_bev_res = ast.literal_eval(config.get("VIGOR", "grd_bev_res"))
    sat_bev_res = config.getint("VIGOR", "sat_bev_res")

    ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
    satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

    grid_size_h = config.getfloat("VIGOR", "grid_size_h") 



    parser = argparse.ArgumentParser()

    parser.add_argument('--random_orientation', choices=('True','False'), default='False')
    parser.add_argument('--ransac', choices=('True','False'), default='False')
    parser.add_argument('--use_filtered', choices=('True','False'), default='False')
    parser.add_argument('--num_epoch', type=int, default=100)
    
    parser.add_argument('--dataset_root', type=str, default=dataset_root)
    parser.add_argument('--label_root', type=str, default=label_root)
    parser.add_argument('--valid_interval', type=int, help='validation exe interval', choices=(1,2), default=1)

    parser.add_argument('--grd_bev_res', type=tuple, default=grd_bev_res, help='Ground BEV resolution')
    parser.add_argument('--sat_bev_res', type=int, default=sat_bev_res, help='Satellite BEV resolution')
    parser.add_argument('--ground_image_size', type=int, nargs=2, default=ground_image_size, help='Ground image size (H, W)')
    parser.add_argument('--satellite_image_size', type=int, nargs=2, default=satellite_image_size, help='Satellite image size (H, W)')
    parser.add_argument('--print_interval', type=float, default=10.0, help='0~1, For example, 0.25 means 4 prints per epoch. if value > 1 not print training condition')

    parser.add_argument('--is_train', type=str, choices=('train','test'), default='train')

    parser.add_argument('--gpuid', type=int, default='2')
    parser.add_argument('--area', type=str, choices=('same','cross'), default='same')
    parser.add_argument('--optim', type=str, choices=('Adam','AdamW'), default='Adam')
    parser.add_argument('--scheduler', type=str, help='scheduler activation', choices=('True','False'), default='False')
    parser.add_argument('--settings', type=str, default='ARC2_v3_gap')
    
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=10)

    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--restore_ckpt',type=str)
    # parser.add_argument('--batch_size', type=int, help='batch size', default=64)
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)

    parser.add_argument('--vce_loss', type=str, help='vce_loss activation', choices=('True','False'), default='True')
    parser.add_argument('--ray_loss', type=str, help='ray_loss activation', choices=('True','False'), default='True')

    args = parser.parse_args()

    global ori_name
    args.random_orientation = args.random_orientation == 'True'
    if args.random_orientation==True:
        ori_name = 'unknown_ori'
    else:
        ori_name = 'known_ori'

    args.ransac = args.ransac == 'True'
    args.vce_loss = args.vce_loss == 'True'
    args.ray_loss = args.ray_loss == 'True'
    args.scheduler = args.scheduler == 'True'
    args.use_filtered = args.use_filtered == 'True'



    NewYork_res = config.getfloat("Constants", "NewYork_res") * 640 / satellite_image_size[0]
    Seattle_res = config.getfloat("Constants", "Seattle_res") * 640 / satellite_image_size[0]
    SanFrancisco_res = config.getfloat("Constants", "SanFrancisco_res") * 640 / satellite_image_size[0]
    Chicago_res = config.getfloat("Constants", "Chicago_res") * 640 / satellite_image_size[0]
    gsd = { # meter per pixel
        'NewYork' : NewYork_res,
        'Chicago' : Chicago_res,
        'SanFrancisco' : SanFrancisco_res,
        'Seattle' : Seattle_res
    }

    print(f"Area: {args.area}, Random orientation: {args.random_orientation}, RANSAC: {args.ransac}")


    print('[--------',f'GPU ID: {args.gpuid}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Batch size: {args.batch_size}'.center(30, ' '), '--------]')
    print('[--------',f'Area: {args.area}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Random orientation: {args.random_orientation}'.center(30, ' '), '--------]')
    print('[--------',f'Ray Loss: {args.ray_loss}'.center(30, ' '), '--------]', end='')
    print('[--------',f'VCE Loss: {args.vce_loss}'.center(30, ' '), '--------]')

    print()
    print()
    print('[========',f'Setting: {args.settings}'.center(30, ' '), '========]', end='')
    print('[========',f'Name: {args.name}'.center(30, ' '), '========]')
    print('[========',f'Sat Grid Resolution: {sat_bev_res}'.center(30, ' '), '========]', end='')
    print('[========',f'Grd Image Size: {ground_image_size}'.center(30, ' '), '========]')
    print()



    if args.is_train=='train':
        print('[ =========== Training start =========== ]')

        train(args, gsd)
    else:
        print('[ =========== Testing start =========== ]')
        print()
        















