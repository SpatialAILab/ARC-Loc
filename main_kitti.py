import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import numpy as np
import math
import random

import torch
import torch.nn as nn

from utils.loss import *
from utils.solver import *
from utils.utils import *

from tqdm import tqdm
import time

from settings import set_loader
from evaluate_func import validation_Arc_KITTI as validation
from dataloaders.dataloader_kitti import train_set, test1_set, test2_set, get_meter_per_pixel
from torch.utils.data import DataLoader




train_loader = DataLoader(train_set, batch_size=16, shuffle=True, pin_memory=True,
                              num_workers=4, drop_last=False)

test1_loader = DataLoader(test1_set, batch_size=16, shuffle=False, pin_memory=True,
                            num_workers=4, drop_last=False)

test2_loader = DataLoader(test2_set, batch_size=16, shuffle=False, pin_memory=True,
                              num_workers=4, drop_last=False)

gsd_kitti = get_meter_per_pixel()


def train(args, gsd):

    device = f'cuda:{args.gpuid}'
    num_samples_matches = args.num_samples


    CVM_model, _, _, optimizer, scheduler, _, current_epoch_ = set_loader(args)
    if args.restore_ckpt is not None:
        print(f'========== Load CKPT : {args.restore_ckpt} ==========')
        current_epoch = current_epoch_
    else:
        current_epoch = 0

    CVM_model.to(device)

    results_dir = os.path.join('results', 'kitti', 'known_ori')
    os.makedirs(results_dir, exist_ok=True)

    # 0. define a metric grid 
    metric_coord4loss = create_metric_grid(grid_size_h, 41, 1).to(device)


    best_dist_mean1 = 1000
    best_dist_mean2 = 1000
    best_dist_median = 1000
    for epoch in range(current_epoch, args.num_epoch):
        CVM_model.train()

        running_loss = 0.0
        running_ray_dist = 0.0
        
        num_batches = 0

        pbar = tqdm(train_loader, dynamic_ncols=True)
        grd_feat_x_size = args.grd_bev_res[1]


        for it, data in enumerate(pbar):
            sat, grd, cam_k, tgt, Rgt, fov_deg = data 

            grd = grd.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)
            cam_k = cam_k.to(device)
            fov_deg = fov_deg.to(device)

            img_width = grd.shape[-1]

            B, _, sat_size, _ = sat.size()
            center = sat_size / 2


            tgt_px = tgt.clone() 
            tgt_px[:,:,0] = center - tgt[:,:,0] # y(h)
            tgt_px[:,:,1] = center - tgt[:,:,1] # x(w)


            # --- 2. grd, sat feature synthesis ---
            matching_score, matching_score_original, other_output = CVM_model(grd, sat, 0)
            B, num_kpts_sat, num_kpts_grd = matching_score_original.shape

            # --- 3. feature descriptor matching ---
            matches_flat = matching_score.flatten(1)
            batch_idx = torch.arange(B).view(B, 1).repeat(1, num_samples_matches).reshape(B, num_samples_matches)
            sampled_idx = torch.multinomial(matches_flat, num_samples_matches)

            sampled_sat_idx = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
            sampled_grd_idx = sampled_idx % num_kpts_grd


            sampled_grd_x_idx = sampled_grd_idx % grd_feat_x_size

            weights = matches_flat[batch_idx, sampled_idx]

            u_g = pano_dirs_from_indices_kitti(sampled_grd_x_idx, grd_feat_x_size, cam_k, img_width) # b, k, 2, dir(x,y)

            # if args.random_orientation:
            u_g = u_g @ Rgt.transpose(1,2) # row vector rotation : CW
            
            pstars, residuals = ARC_Solver(
                sampled_sat_idx=sampled_sat_idx,   # (B,K)
                u_ground=u_g,                 # r(B,K,2)
                Hs=sat_size,
                sat_bev_res=sat_bev_res,
                weights=weights,               # (B,K) 또는 None
            )

            trans = pstars.clone() # pstars : pixel level coord
            # pixel to translation
            trans[:,0] = center - pstars[:,1] 
            trans[:,1] = center - pstars[:,0]
            trans = trans.unsqueeze(1) # trans : meter level translation

            t = trans/sat_size * grid_size_h # pixel level translation >> meter level translation?

            # --- 4. loss function & gradient ---
            losses = []

            if args.vce_loss :
                loss_vce = compute_vce_loss(metric_coord4loss, Rgt, t, Rgt, tgt*grid_size_h/sat_size)
                losses.append(loss_vce.mean())

            if args.ray_loss: 

                loss_ray_dist = residuals.mean()
                losses.append(args.alpha*loss_ray_dist)

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

    
        t_error1, t_median_error2 = validation(args, CVM_model, test1_loader, results_dir, device, gsd)
        t_error2, t_median_error2 = validation(args, CVM_model, test2_loader, results_dir, device, gsd)
            
        if t_error1 < best_dist_mean1:
            best_dist_mean1 = t_error1

            save_path = f'checkpoints/kitti/{ori_name}/' + f'Best_test1_ckpt_{args.settings}_{args.name}.pth'
            save_ckpts(
                save_path = save_path,
                CVM_model=CVM_model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_epoch=epoch,
                )
        if t_error2 < best_dist_mean2:
            best_dist_mean2 = t_error2

            save_path = f'checkpoints/kitti/{ori_name}/' + f'Best_test2_ckpt_{args.settings}_{args.name}.pth'
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


def test(args, gsd):
    from evaluate_func import test_Arc_KITTI as validation

    device = f'cuda:{args.gpuid}'
    CVM_model, _ = set_loader(args)
    CVM_model.to(device)

    results_dir = 'results/kitti/'

    # 테스트 데이터로더 선택
    loader = test1_loader if args.test_set == 'test1' else test2_loader
    area_name = "Same" if args.test_set == 'test1' else "Cross"

    print(f'[ =========== Testing start: {area_name} Area =========== ]')
    
    # 4개 에러 배열 반환 (validation 함수 내부에서 pstars -> trans 변환이 완료된 상태여야 함)
    t_err, y_err, long_err, lat_err = validation(args, CVM_model, loader, gsd, device)

    # numpy 변환
    t_err, y_err = np.array(t_err), np.array(y_err)
    long_err, lat_err = np.array(long_err), np.array(lat_err)

    # Recall@N 계산 함수
    def R(arr, thresh): return np.mean(arr < thresh) * 100

    # 테이블 형식에 맞춘 출력
    print("\n" + "="*100)
    print(f"{'Methods':<12} | {'Loc. (m)':^15} | {'Lateral (%)':^15} | {'Long. (%)':^15} | {'Orien. (deg)':^15} | {'Orien. (%)':^15}")
    print(f"{'':<12} | {'Mean':^7} {'Med.':^7} | {'R@1m':^7} {'R@5m':^7} | {'R@1m':^7} {'R@5m':^7} | {'Mean':^7} {'Med.':^7} | {'R@1°':^7} {'R@5°':^7}")
    print("-" * 100)
    
    # 현재 모델의 결과값 (ARC-Pose 또는 현재 실험 중인 모델명)
    method_name = args.name if hasattr(args, 'name') else "MyModel"
    
    print(f"{method_name:<12} | "
          f"{np.mean(t_err):.2f} {np.median(t_err):.2f} | "  # Loc. (m)
          f"{R(lat_err, 1):.2f} {R(lat_err, 5):.2f} | "      # Lateral (%)
          f"{R(long_err, 1):.2f} {R(long_err, 5):.2f} | "    # Long. (%)
          f"{np.mean(y_err):.2f} {np.median(y_err):.2f} | "  # Orien. (deg)
          f"{R(y_err, 1):.2f} {R(y_err, 5):.2f}")            # Orien. (%)
    print("="*100)

    # 파일 저장 (Latex Table 형식으로 기록해두면 논문 쓰기 편합니다)
    results_dir = os.path.join('results', 'kitti')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'table_format.txt'), 'a') as f:
        f.write("==================================================================\n")
        f.write(f"\n{area_name} - {method_name}\n")
        f.write(f"{np.mean(t_err):.2f} & {np.median(t_err):.2f} & "
                f"{R(lat_err, 1):.2f} & {R(lat_err, 5):.2f} & "
                f"{R(long_err, 1):.2f} & {R(long_err, 5):.2f} & "
                f"{np.mean(y_err):.2f} & {np.median(y_err):.2f} & "
                f"{R(y_err, 1):.2f} & {R(y_err, 5):.2f} \\\\\n")


                

import torch
import math


if __name__ == '__main__':
    import configparser
    import ast

    import sys
    print("--- 환경 진단 시작 ---")
    print(f"파이썬 실행 파일 경로: {sys.executable}")
    print(f"실행된 PyTorch 버전: {torch.__version__}")
    print(f"실행된 PyTorch 위치: {torch.__file__}")
    print("--- 환경 진단 끝 ---")

    seed = 777
    print('seed number:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(mode=True, warn_only=True)

    # Load configuration

    config = configparser.ConfigParser()
    config.read("./config.ini")


    grd_bev_res = ast.literal_eval(config.get("KITTI", "grd_bev_res"))
    sat_bev_res = ast.literal_eval(config.get("KITTI", "sat_bev_res"))
    
    # num_samples_matches = config.getint("VIGOR", "num_samples_matches")

    ground_image_size = ast.literal_eval(config.get("KITTI", "ground_image_size"))
    satellite_image_size = ast.literal_eval(config.get("KITTI", "satellite_image_size"))

    grid_size_h = config.getfloat("KITTI", "grid_size_h") 
    grid_size_v = config.getfloat("KITTI", "grid_size_v") 



    parser = argparse.ArgumentParser()

    parser.add_argument('--random_orientation', choices=('True','False'), default='False')
    parser.add_argument('--ransac', choices=('True','False'), default='False')
    parser.add_argument('--num_epoch', type=int, default=100)
    
    parser.add_argument('--valid_interval', type=int, help='validation exe interval', choices=(1,2), default=1)

    parser.add_argument('--grd_bev_res', type=int, default=grd_bev_res, help='Ground BEV resolution')
    parser.add_argument('--sat_bev_res', type=int, default=sat_bev_res, help='Satellite BEV resolution')
    parser.add_argument('--ground_image_size', type=int, nargs=2, default=ground_image_size, help='Ground image size (H, W)')
    parser.add_argument('--satellite_image_size', type=int, nargs=2, default=satellite_image_size, help='Satellite image size (H, W)')
    parser.add_argument('--print_interval', type=float, default=10.0, help='0~1, For example, 0.25 means 4 prints per epoch. if value > 1 not print training condition')

    parser.add_argument('--is_train', type=str, choices=('train','test'), default='test')

    parser.add_argument('--gpuid', type=int, default='3')
    parser.add_argument('--area', type=str, choices=('same','cross'), default='same')
    parser.add_argument('--optim', type=str, choices=('Adam','AdamW'), default='Adam')
    parser.add_argument('--scheduler', type=str, help='scheduler activation', choices=('True','False'), default='False')
    parser.add_argument('--settings', type=str, default='ARC2_v3_kitti')
    
    parser.add_argument('--num_samples', type=int, default=512)
    parser.add_argument('--lr', default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=10)

    parser.add_argument('--name', type=str, default='debug')
    # parser.add_argument('--restore_ckpt',default='checkpoints/kitti/known_ori/Best_test1_ckpt_ARC2_v3_kitti_fix_width_code.pth', type=str)
    parser.add_argument('--restore_ckpt', type=str)
    parser.add_argument('--test_set', default='test1', type=str)
    parser.add_argument('--batch_size', type=int, help='batch size', default=4)
    parser.add_argument('--w_scaler', type=int, help='consistency loss w scale value (before softmax)', default=1)

    parser.add_argument('--vce_loss', type=str, help='vce_loss activation', choices=('True','False'), default='True')
    parser.add_argument('--ray_loss', type=str, help='ray_dist_loss activation', choices=('True','False'), default='True')
    parser.add_argument('--ray_dir_loss', type=str, help='ray_dir_loss activation', choices=('True','False'), default='False')

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
    args.ray_dir_loss = args.ray_dir_loss == 'True'
    args.scheduler = args.scheduler == 'True'


    print(f" Random orientation: {args.random_orientation}, RANSAC: {args.ransac}")
    # os.makedirs(args.name, exist_ok=True)


    print('[--------',f'GPU ID: {args.gpuid}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Batch size: {args.batch_size}'.center(30, ' '), '--------]')
    print('[--------',f'Area: Same, Cross'.center(30, ' '), '--------]', end='')
    print('[--------',f'Random orientation: {args.random_orientation}'.center(30, ' '), '--------]')
    print('[--------',f'Ray Dist Loss: {args.ray_loss}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Ray Dir Loss: {args.ray_dir_loss}'.center(30, ' '), '--------]')
    print('[--------',f'Pose Loss: {args.vce_loss}'.center(30, ' '), '--------]')

    print()
    print()
    print('[========',f'Setting: {args.settings}'.center(30, ' '), '========]', end='')
    print('[========',f'Name: {args.name}'.center(30, ' '), '========]')
    print('[========',f'Sat Grid Resolution: {sat_bev_res}'.center(30, ' '), '========]', end='')
    print('[========',f'Grd Image Size: {ground_image_size}'.center(30, ' '), '========]')
    print()




    if args.is_train=='train':
        print('Training start')
        args.restore_ckpt=None
        args.ransac = False
        train(args, gsd_kitti)
    else: # test
        print('[ =========== Testing start =========== ]')
        test(args, gsd_kitti)
        















