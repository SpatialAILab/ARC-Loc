import os

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import numpy as np

from tqdm import tqdm
import torch
from utils.utils import ARCSolverRANSAC
from utils.solver import *
from datetime import datetime

import ast
import configparser



config = configparser.ConfigParser()
config.read("./config.ini")
# config.read("./config_dino.ini")

grd_height_res = config.getint("VIGOR", "grd_height_res")
num_samples_matches = config.getint("VIGOR", "num_samples_matches")

ground_image_size = ast.literal_eval(config.get("KITTI", "ground_image_size"))
satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

grid_size_h = config.getfloat("VIGOR", "grid_size_h") 
grid_size_v = config.getfloat("VIGOR", "grid_size_v") 


th_soft_inlier = config.getfloat("VIGOR", "th_soft_inlier") 
th_inlier = config.getfloat("VIGOR", "th_inlier") 
num_samples_matches_ransac = config.getint("VIGOR", "num_samples_matches_ransac")
num_corr_2d_2d = config.getint("VIGOR", "num_corr_2d_2d")
it_matches = config.getint("VIGOR", "it_matches")
it_RANSAC_procrustes = config.getint("VIGOR", "it_RANSAC_procrustes")
num_ref_steps = config.getint("VIGOR", "num_ref_steps")

NewYork_res = config.getfloat("Constants", "NewYork_res") * 640 / satellite_image_size[0]
Seattle_res = config.getfloat("Constants", "Seattle_res") * 640 / satellite_image_size[0]
SanFrancisco_res = config.getfloat("Constants", "SanFrancisco_res") * 640 / satellite_image_size[0]
Chicago_res = config.getfloat("Constants", "Chicago_res") * 640 / satellite_image_size[0]


def validation_Arc(args, CVM_model, valid_loader, results_dir, device):
    global w_scaler
    w_scaler = args.w_scaler
    
    '''
    RANSAC
    '''
    CVM_model.eval()
    num_samples_matches = args.num_samples
    sat_bev_res = args.sat_bev_res
    residuals=np.array([0.0])
    weights_mean = []
    with torch.no_grad():
        # translation_error = []
        translation_line_error = []
        avg_dist_error = []
        city_error_dict = {
            'NewYork': [],
            'Seattle': [],
            'SanFrancisco': [],
            'Chicago': []
        }
        
        prev= 'NewYork'
        # max_iter = 300
        pbar = tqdm(valid_loader, desc="Validating")
        for i, data in enumerate(pbar):
            # if i >= max_iter:
            #     break

            grd, sat, tgt, Rgt, city, remains = data
            pseudo_epoch = 50
            remains.append(pseudo_epoch)

            grd = grd.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)

            B, _, sat_size, _ = sat.size()


            matching_score, matching_score_orig, _ = CVM_model(grd, sat, remains)
            if matching_score.shape[0] > B:
                matching_score = matching_score[:B]
            center = sat_size/2

            B, num_kpts_sat, num_kpts_grd = matching_score.shape
            if args.ransac:
                e2e_ARC_LS = ARCSolverRANSAC(
                                    it_RANSAC=it_RANSAC_procrustes,         # RANSAC 반복 횟수
                                    it_matches=it_matches,                   # 매칭 샘플링 횟수
                                    num_samples_matches=num_samples_matches_ransac,   # RANSAC에 사용할 총 매칭 수
                                    num_corr_lines=num_corr_2d_2d,           # 최소 샘플 라인 수 (K)
                                    num_refinements=num_ref_steps,           # 리핏 횟수
                                    th_inlier=th_inlier,                     # 하드 인라이어 임계값 (픽셀)
                                    th_soft_inlier=th_soft_inlier,           # 소프트 인라이어 임계값 (픽셀)
                                    Hs=sat_size, Ws=sat_size,                             # 위성 이미지 너비 (정방형 가정)
                                    grd_bev_res=args.grd_bev_res,
                                    sat_bev_res=args.sat_bev_res,
                                )
                # '''
                R, pstars, best_inliers, inliers = e2e_ARC_LS.estimate_pose(
                    # matching_score*100000,
                    matching_score,
                    matching_score_orig, 
                    return_inliers=False
                )


            else:
                matches_row = matching_score.flatten(1)
                batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
                sampled_idx = torch.multinomial(matches_row, num_samples_matches)
            
                sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
                sampled_idx_grd = (sampled_idx % num_kpts_grd)
            
                grd_feat_x_size = 2*int((num_kpts_grd//2)**(1/2)) # feature w size

                sampled_grd_x_idx = sampled_idx_grd % grd_feat_x_size

                weights = matches_row[batch_idx, sampled_idx]
                

                u_g = pano_dirs_from_indices(sampled_grd_x_idx, grd_feat_x_size) # b, k, 2, dir(ux,uy)

                if args.random_orientation:
                    u_g =  u_g @ Rgt

                pstars, residuals = ARC_Solver(
                    sampled_sat_idx=sampled_idx_sat,   # (B,K)
                    u_ground=u_g,                 # (B,K,2)
                    Hs=sat_size,
                    sat_bev_res=sat_bev_res,
                    weights=weights,               # (B,K) 또는 None
                )

            # pixel coordinates to translation vector
            trans = pstars.clone()
            trans[:,0] = center - pstars[:,1] 
            trans[:,1] = center - pstars[:,0]

            offset_line = torch.abs(trans.unsqueeze(1).to(tgt.device)-tgt).cpu().detach().numpy()
            translation_error_B_line = np.sqrt(offset_line[:,0,0]**2 + offset_line[:,0,1]**2)

            avg_dist_error.append(residuals.mean().item())

            Rgt = Rgt.cpu().detach().numpy()
            for b in range(B):
                if city[b] == 'NewYork':

                    err_line = translation_error_B_line[b] * NewYork_res
                    translation_line_error.append(err_line)
                    city_error_dict['NewYork'].append(err_line)

                elif city[b] == 'Seattle':
                    
                    err_line = translation_error_B_line[b] * Seattle_res
                    translation_line_error.append(err_line)
                    city_error_dict['Seattle'].append(err_line)

                elif city[b] == 'SanFrancisco':

                    err_line = translation_error_B_line[b] * SanFrancisco_res
                    translation_line_error.append(err_line)
                    city_error_dict['SanFrancisco'].append(err_line)

                    
                elif city[b] == 'Chicago':

                    err_line = translation_error_B_line[b] * Chicago_res
                    translation_line_error.append(err_line)
                    city_error_dict['Chicago'].append(err_line)

            current_mean_line = np.mean(translation_line_error)
            p2l_dist = np.mean(avg_dist_error)
            pbar.set_postfix({
                'trans_err_line_mean': f"{current_mean_line:.2f}",
                'p2l_dist_mean': f"{p2l_dist:.2f}",
            })

        translation_error_mean_line = np.mean(translation_line_error)
        translation_error_median_line = np.median(translation_line_error)
        avg_dist_error_mean = np.mean(avg_dist_error)
        

        # print()
        print('[--------',f'translation error Line mean: {translation_error_mean_line}'.center(30, ' '), '--------]')
        print('[--------',f'translation error Line median: {translation_error_median_line}'.center(30, ' '), '--------]')

        for k, v in city_error_dict.items():
            if len(v) > 0:
                print(f"[{k}] mean: {np.mean(v):.2f}, median: {np.median(v):.2f}, N={len(v)}")
            else:
                print(f"[{k}] no samples")



        with open(os.path.join(results_dir, f'{args.settings}_{args.name}_{args.is_train}_results.txt'), 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], {avg_dist_error_mean}\n")
            f.write(f'translation_error_mean_line: {translation_error_mean_line:.6f}\n')
            f.write(f'translation_error_median_line: {translation_error_median_line:.6f}\n')
            if args.is_train=='test':
                for k, v in city_error_dict.items():
                    if len(v) > 0:
                        f.write(f"[{k}] mean: {np.mean(v):.2f}, median: {np.median(v):.2f}, N={len(v)}\n")
                    else:
                        f.write(f"[{k}] no samples\n")
    CVM_model.train()
    return translation_error_mean_line, translation_error_median_line










def validation_Arc_KITTI(args, CVM_model, valid_loader, results_dir, device, gsd):
    
    grid_size_h = config.getfloat("KITTI", "grid_size_h") 
    grid_size_v = config.getfloat("KITTI", "grid_size_v") 


    '''
    RANSAC
    '''
    CVM_model.eval()
    num_samples_matches = args.num_samples
    sat_bev_res = args.sat_bev_res
    residuals=np.array([0.0])
    weights_mean = []
    with torch.no_grad():
        # translation_error = []
        translation_line_error = []
        avg_dist_error = []
        # yaw_error = []
        # yaw_pred_list = []
        # yaw_gt_list = []

        city_error_dict = {
            'NewYork': [],
            'Seattle': [],
            'SanFrancisco': [],
            'Chicago': []
        }
        
        prev= 'NewYork'
        # max_iter = 300
        pbar = tqdm(valid_loader, desc="Validating")
        for i, data in enumerate(pbar):
            # if i >= max_iter:
            #     break

            sat, grd, cam_k, tgt, Rgt, fov_deg = data 
            pseudo_epoch = 50

            grd = grd.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)
            cam_k = cam_k.to(device)
            fov_deg = fov_deg.to(device)
            
            B, _, sat_size, _ = sat.size()

            img_width = grd.shape[-1]

            matching_score, matching_score_orig, _ = CVM_model(grd, sat, 0)
            if matching_score.shape[0] > B:
                matching_score = matching_score[:B]
            center = sat_size/2

            B, num_kpts_sat, num_kpts_grd = matching_score.shape
            if args.ransac:
                e2e_BearingLS = e2eUnknownOriRANSAC_Vectorized_KITTI(
                                                                yaw_deg_step=1.0,
                                                                yaw_full360=False,          # 360도 스캔
                                                                # yaw_full360=False,
                                                                yaw_candidates_deg=None,   # None이면 step 기반으로 자동 생성
                                                                # yaw_candidates_deg=list(range(-45, 46, 1)),
                                                                # stage-1
                                                                H_yaw=24, M_yaw=6, N_yaw=2048,
                                                                # stage-2 (기존 파라미터 유지)
                                                                it_RANSAC=it_RANSAC_procrustes,
                                                                it_matches=it_matches,
                                                                num_samples_matches=num_samples_matches_ransac,
                                                                num_corr_lines=num_corr_2d_2d,
                                                                num_refinements=num_ref_steps,
                                                                th_inlier=th_inlier,
                                                                th_soft_inlier=th_soft_inlier,
                                                                Hs=sat_size, Ws=sat_size,
                                                                grd_bev_res=args.grd_bev_res,
                                                                sat_bev_res=args.sat_bev_res,
                                                                disamb_beta = 6.0,
                                                                disamb_scale = 80.0,
                                                            )
                # '''
                R, pstars, best_inliers, A = e2e_BearingLS.estimate_pose(
                    matching_score,
                    matching_score_orig, 
                    cam_k,
                    return_inliers=False
                )
                trans = pstars.clone()
                trans[:,0] = center - pstars[:,1] 
                trans[:,1] = center - pstars[:,0]

                R = -R  # ArcSolver와 좌표계 맞추기 위해 부호 반전
            else:
                matches_row = matching_score.flatten(1)
                batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
                sampled_idx = torch.multinomial(matches_row, num_samples_matches)
            
                sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
                sampled_idx_grd = (sampled_idx % num_kpts_grd)


                grd_feat_x_size = args.grd_bev_res[1]

                sampled_grd_x_idx = sampled_idx_grd % grd_feat_x_size

                weights = matches_row[batch_idx, sampled_idx]
                

                u_g = pano_dirs_from_indices_kitti(sampled_grd_x_idx, grd_feat_x_size, cam_k, img_width) # b, k, 2, dir(ux,uy)


                u_g = u_g @ Rgt # kitti +-10 ori noise

                pstars, residuals = ARC_Solver(
                    sampled_sat_idx=sampled_idx_sat,   # (B,K)
                    u_ground=u_g,                 # (B,K,2)
                    Hs=sat_size,
                    sat_bev_res=sat_bev_res,
                    weights=weights,               # (B,K) 또는 None
                )
            
            # pixel coordinates to translation vector
            trans = pstars.clone()
            trans[:,0] = center - pstars[:,1] 
            trans[:,1] = center - pstars[:,0]

            offset_line = torch.abs(trans.unsqueeze(1).to(tgt.device)-tgt).cpu().detach().numpy()
            translation_error_B_line = np.sqrt(offset_line[:,0,0]**2 + offset_line[:,0,1]**2)

            avg_dist_error.append(residuals.mean().item())

            Rgt = Rgt.cpu().detach().numpy()
            for b in range(B):

                err_line = translation_error_B_line[b] * gsd
                translation_line_error.append(err_line)

            current_mean_line = np.mean(translation_line_error)
            p2l_dist = np.mean(avg_dist_error)
            pbar.set_postfix({
                'trans_err_line_mean': f"{current_mean_line:.2f}",
                'p2l_dist_mean': f"{p2l_dist:.2f}",
            })

        translation_error_mean_line = np.mean(translation_line_error)
        translation_error_median_line = np.median(translation_line_error)
        avg_dist_error_mean = np.mean(avg_dist_error)
        

        # print('[========',f'Valid/Test dataset inference'.center(40, ' '), '========]')
        # print()
        print('[--------',f'translation error Line mean: {translation_error_mean_line}'.center(30, ' '), '--------]')
        print('[--------',f'translation error Line median: {translation_error_median_line}'.center(30, ' '), '--------]')

        with open(os.path.join(results_dir, f'{args.settings}_{args.name}_{args.is_train}_results.txt'), 'a') as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}], {avg_dist_error_mean}\n")
            f.write(f'translation_error_mean_line: {translation_error_mean_line:.6f}\n')
            f.write(f'translation_error_median_line: {translation_error_median_line:.6f}\n')

    CVM_model.train()
    return translation_error_mean_line, translation_error_median_line



def test_Arc_KITTI(args, CVM_model, valid_loader, gsd, device):
    global w_scaler
    w_scaler = args.w_scaler
    th_soft_inlier = config.getfloat("KITTI", "th_soft_inlier") 
    th_inlier = config.getfloat("KITTI", "th_inlier") 
    num_samples_matches_ransac = config.getint("KITTI", "num_samples_matches_ransac")
    num_corr_2d_2d = config.getint("KITTI", "num_corr_2d_2d")
    it_matches = config.getint("KITTI", "it_matches")
    it_RANSAC_procrustes = config.getint("KITTI", "it_RANSAC_procrustes")
    num_ref_steps = config.getint("KITTI", "num_ref_steps")
    '''
    RANSAC
    '''
    CVM_model.eval()
    num_samples_matches = args.num_samples
    sat_bev_res = args.sat_bev_res
    with torch.no_grad():
        translation_error = []
        longitudinal_error = []
        lateral_error = []
        yaw_error = []
        
        prev= 'NewYork'
        # max_iter = 300
        pbar = tqdm(valid_loader, desc="Validating")
        for i, data in enumerate(pbar):
            # if i >= max_iter:
            #     break

            sat, grd, cam_k, tgt, Rgt, fov_deg = data 
            pseudo_epoch = 50

            grd = grd.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)
            cam_k = cam_k.to(device)
            fov_deg = fov_deg.to(device)
            
            B, _, sat_size, _ = sat.size()

            img_width = grd.shape[-1]

            matching_score, matching_score_orig, _ = CVM_model(grd, sat, 0)
            if matching_score.shape[0] > B:
                matching_score = matching_score[:B]
            center = sat_size/2

            B, num_kpts_sat, num_kpts_grd = matching_score.shape
            if args.ransac:
                # e2e_ARC_LS = e2eBearingLSSolverRANSAC_Vectorized(
                #                     it_RANSAC=it_RANSAC_procrustes,         # RANSAC 반복 횟수
                #                     it_matches=it_matches,                   # 매칭 샘플링 횟수
                #                     num_samples_matches=num_samples_matches_ransac,   # RANSAC에 사용할 총 매칭 수
                #                     num_corr_lines=num_corr_2d_2d,           # 최소 샘플 라인 수 (K)
                #                     num_refinements=num_ref_steps,           # 리핏 횟수
                #                     th_inlier=th_inlier,                     # 하드 인라이어 임계값 (픽셀)
                #                     th_soft_inlier=th_soft_inlier,           # 소프트 인라이어 임계값 (픽셀)
                #                     Hs=sat_size, Ws=sat_size,                             # 위성 이미지 너비 (정방형 가정)
                #                     grd_bev_res=args.grd_bev_res,
                #                     sat_bev_res=args.sat_bev_res,
                #                     # rgt=Rgt
                #                 )
                # # '''
                # R, pstars, best_inliers, inliers = e2e_ARC_LS.estimate_pose(
                #     # matching_score*100000,
                #     matching_score,
                #     matching_score_orig, 
                #     return_inliers=False
                # )
                ori_noise = 10.0
                lo = int(round(-2.0 * ori_noise))
                hi = int(round(2.0 * ori_noise))
                yaw_candidates = [k / 2.0 for k in range(lo, hi+1)]


                e2e_BearingLS = e2eUnknownOri_KITTI(
                                                yaw_deg_step=1,
                                                yaw_full360=False,          # 360도 스캔
                                                # yaw_full360=False,
                                                yaw_candidates_deg=yaw_candidates,   # None이면 step 기반으로 자동 생성
                                                # yaw_candidates_deg=list(range(-45, 46, 1)),
                                                # stage-1
                                                H_yaw=512, M_yaw=6, N_yaw=1024,
                                                # stage-2 (기존 파라미터 유지)
                                                it_RANSAC=it_RANSAC_procrustes,
                                                it_matches=it_matches,
                                                # num_samples_matches=num_samples_matches_ransac,
                                                num_samples_matches=2048,
                                                num_corr_lines=num_corr_2d_2d,
                                                num_refinements=num_ref_steps,
                                                th_inlier=th_inlier,
                                                th_soft_inlier=th_soft_inlier,
                                                Hs=sat_size, Ws=sat_size,
                                                grd_bev_res=args.grd_bev_res,
                                                sat_bev_res=args.sat_bev_res,
                                                disamb_beta = 6.0,
                                                disamb_scale = 80.0,
                                            )

                R, pstars, best_inliers= e2e_BearingLS.estimate_pose(
                    matching_score,
                    matching_score_orig, 
                    cam_k,
                    return_inliers=False
                )
                trans = pstars.clone()
                trans[:,0] = center - pstars[:,1] 
                trans[:,1] = center - pstars[:,0]
                R = -R  # ArcSolver와 좌표계 맞추기 위해 부호 반전 (코드상에서 R이 아닌 yaw값을 출력)

            else:

                matches_row = matching_score.flatten(1)
                batch_idx = torch.tile(torch.arange(B).view(B, 1), [1, num_samples_matches]).reshape(B, num_samples_matches)
                sampled_idx = torch.multinomial(matches_row, num_samples_matches)
            
                sampled_idx_sat = torch.div(sampled_idx, num_kpts_grd, rounding_mode='trunc')
                sampled_idx_grd = (sampled_idx % num_kpts_grd)
            
                grd_feat_x_size = args.grd_bev_res[1]

                sampled_grd_x_idx = sampled_idx_grd % grd_feat_x_size

                weights = matches_row[batch_idx, sampled_idx]
                

                u_g = pano_dirs_from_indices_kitti(sampled_grd_x_idx, grd_feat_x_size, cam_k, img_width) # b, k, 2, dir(ux,uy)


                pstars, residuals = ARC_Solver(
                    sampled_sat_idx=sampled_idx_sat,   # (B,K)
                    u_ground=u_g,                 # (B,K,2)
                    Hs=sat_size,
                    sat_bev_res=sat_bev_res,
                    weights=weights,               # (B,K) 또는 None
                )
                # R=Rgt



            # pixel coordinates to translation vector
            trans = pstars.clone()
            trans[:,0] = center - pstars[:,1] 
            trans[:,1] = center - pstars[:,0]
            # trans : pixel unit vector
            trans = trans.unsqueeze(1).cpu()
            tgt = tgt.cpu()


            R = R.cpu()
            Rgt = Rgt.cpu()
            

            for b in range(B):
                loc_pred = [sat_size / 2 - trans[b, 0, 1], sat_size / 2 - trans[b, 0, 0]]
                loc_gt = [sat_size / 2 - tgt[b, 0, 1], sat_size / 2 - tgt[b, 0, 0]]

                distance = np.sqrt((loc_gt[0]-loc_pred[0])**2+(loc_gt[1]-loc_pred[1])**2) * gsd
                translation_error.append(distance)
                
                yaw = R[b]/ np.pi * 180
                
                cos_gt = Rgt[b,0,0]
                sin_gt = Rgt[b,1,0]
                yaw_gt = np.arctan2(sin_gt, cos_gt) / np.pi * 180
                diff = np.abs(yaw - yaw_gt)
                
                yaw_error.append(np.min([diff, 360-diff]))
                
                e = np.array(loc_pred, dtype=float) - np.array(loc_gt, dtype=float)
    
                # (0=up, 90=left)
                theta = np.deg2rad(-yaw_gt)
                
                # Unit vectors tied to GT heading:
                # forward (longitudinal) and left-normal (lateral)
                u_long = np.array([-np.sin(theta),  np.cos(theta)])   # forward
                u_lat  = np.array([-np.cos(theta), -np.sin(theta)])   # left
                
                # Project to get components (in meters)
                err_longitudinal = np.abs(float(e @ u_long) * gsd)
                err_lateral      = np.abs(float(e @ u_lat)  * gsd)
                
                longitudinal_error.append(err_longitudinal)
                lateral_error.append(err_lateral)
    
        return np.array(translation_error), np.array(yaw_error), np.array(longitudinal_error), np.array(lateral_error)
