import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import argparse
import numpy as np
import random

import torch
from settings import set_loader

def test(args):

    from evaluate_func import validation_Arc as validation


    device = f'cuda:{args.gpuid}'

    CVM_model, test_loader = set_loader(args)
    CVM_model.to(device)

    results_dir = os.path.join('results', 'vigor', args.area, 'known_ori')
    os.makedirs(results_dir, exist_ok=True)

    print('[ =========== Testing start =========== ]')
    t_error = validation(args, CVM_model, test_loader, results_dir, device)






if __name__ == '__main__':
    import configparser
    import ast

    seed = 777
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


    # Load configuration

    config = configparser.ConfigParser()
    config.read("./config.ini")

    dataset_root = config["VIGOR"]["dataset_root"]
    label_root = config["VIGOR"]["label_root"]

    grd_bev_res = ast.literal_eval(config.get("VIGOR", "grd_bev_res"))
    grd_height_res = config.getint("VIGOR", "grd_height_res")
    sat_bev_res = config.getint("VIGOR", "sat_bev_res")

    ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
    satellite_image_size = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

    grid_size_h = config.getfloat("VIGOR", "grid_size_h") 
    grid_size_v = config.getfloat("VIGOR", "grid_size_v") 



    parser = argparse.ArgumentParser()

    parser.add_argument('--is_train', type=str, default='test')

    parser.add_argument('--random_orientation', choices=('True','False'), default='False')
    parser.add_argument('--ransac', choices=('True','False'), default='False')

    parser.add_argument('--dataset_root', type=str, default=dataset_root)
    parser.add_argument('--label_root', type=str, default=label_root)

    parser.add_argument('--grd_bev_res', type=tuple, default=grd_bev_res, help='Ground BEV resolution')
    parser.add_argument('--grd_height_res', type=int, default=grd_height_res, help='Ground height resolution')
    parser.add_argument('--sat_bev_res', type=int, default=sat_bev_res, help='Satellite BEV resolution')
    # parser.add_argument('--num_samples_matches', type=int, default=num_samples_matches, help='Number of matching points to sample')
    parser.add_argument('--ground_image_size', type=int, nargs=2, default=ground_image_size, help='Ground image size (H, W)')
    parser.add_argument('--satellite_image_size', type=int, nargs=2, default=satellite_image_size, help='Satellite image size (H, W)')


    parser.add_argument('--gpuid', type=int, default='3')
    parser.add_argument('--area', type=str, choices=('same','cross'), default='same')
    parser.add_argument('--settings', type=str, default='ARC_Loc_baseline')
    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of matching points to sample')

    parser.add_argument('--batch_size', type=int, help='batchsize', default=16)
    parser.add_argument('--restore_ckpt',type=str)
    parser.add_argument('--use_filtered', choices=('True','False'), default='False')
    

    args = parser.parse_args()

    args.random_orientation = args.random_orientation == 'True'
    args.ransac = args.ransac == 'True'
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
    print()
    print()
    print(f" Area: {args.area}, Random orientation: {args.random_orientation}, RANSAC: {args.ransac}")
    print()
    print()
    # os.makedirs(args.name, exist_ok=True)

    print('[ =========== Testing start =========== ]')
    print('[========',f'CKPT: {args.restore_ckpt}'.center(30, ' '), '========]')
    print()


    print('[--------',f'GPU ID: {args.gpuid}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Batch size: {args.batch_size}'.center(30, ' '), '--------]')
    print('[--------',f'Area: {args.area}'.center(30, ' '), '--------]', end='')
    print('[--------',f'Random orientation: {args.random_orientation}'.center(30, ' '), '--------]')
    print()
    print('[========',f'Setting: {args.settings}'.center(30, ' '), '========]', end='')
    print('[========',f'Name: {args.name}'.center(30, ' '), '========]')
    print('[========',f'Grd Image Size: {ground_image_size}'.center(30, ' '), '========]')

    print()

    print()
    test(args)
        
