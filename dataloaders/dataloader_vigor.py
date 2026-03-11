import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Load configuration
import configparser
import ast
config = configparser.ConfigParser()
config.read("./config.ini")
# config.read("./config_dino.ini")

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


GROUND_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
SATELLITE_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))


# Define transformations
transform_grd = transforms.Compose([
    transforms.Resize(GROUND_IMAGE_SIZE),
    transforms.ToTensor(),
])

transform_sat = transforms.Compose([
    transforms.Resize(SATELLITE_IMAGE_SIZE),
    transforms.ToTensor(),
    ])


def fetch_dataloader_VIGOR(args, split='train'):

    # train_dataset = VIGORDataset(args, split)
    train_dataset = VIGORDataset(root=args.dataset_root, label_root=args.label_root, split=args.area, train=split, \
                                transform=(transform_grd, transform_sat), random_orientation=args.random_orientation)
    print('Training with %d image pairs' % len(train_dataset))

    if split == 'train':
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        print("using {} images for training, {} images for validation.".format(train_size, val_size))
        nw = 24
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=nw)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=nw)

        return train_loader, val_loader
    
    else: # test time
        nw = 24  # number of workers
        print('Using {} dataloader workers every process'.format(nw)) 
        test_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                pin_memory=True, shuffle=False, num_workers=nw, drop_last=False)    
        
        return test_loader




class VIGORDataset(Dataset):
    def __init__(self, root='/home/ziminxia/Work/datasets/VIGOR', label_root=None, split=None, train=None, random_orientation=None, transform=None, first_run=None):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.random_orientation = random_orientation
        self.first_run = first_run
        self.grdimage_transform, self.satimage_transform = (transform_grd, transform_sat)

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()
        self.data_size = len(self.grd_list)

        if random_orientation:
            if not first_run:
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_pred.txt')) as file:
                    self.ori_pred = [line.rstrip() for line in file]
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_gt.txt')) as file:
                    self.ori_gt = [line.rstrip() for line in file]
    

    def _get_city_list(self):
        if self.split == 'samearea':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'crossarea':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []

    def _load_satellite_data(self):
        sat_list, sat_index_dict = [], {}
        idx = 0
        
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            print(f'Loaded {sat_list_fname}, {idx} entries')
        
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        sat_cover_dict = {}
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)
            
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    
                    if label[0] not in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            
            print(f'Loaded {label_fname}, {idx} entries')
        
        return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

    def _get_label_file(self, city):
        if self.split == 'samearea':
            return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
        return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
        grd = self.grdimage_transform(grd)
        


        if self.random_orientation:
            if self.first_run:
                rotation = np.random.uniform(-180/360, 180/360) # -0.5 ~ 0.5
            else:
                if self.ori_pred[idx] == 'None':
                    rotation = np.random.uniform(-180/360, 180/360) 
                else:
                    rotation = (float(self.ori_gt[idx]) - float(self.ori_pred[idx])) / 360
        else:
            rotation = 0



        grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
        yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

        sat, row_offset, col_offset = self._load_satellite_image(idx)

        gt_loc = torch.tensor([[-row_offset, col_offset]])
        r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
        city = self._get_city_name(self.grd_list[idx])
        
        return grd_rolled, sat, gt_loc, r, city

    def _load_image(self, path, default_size):
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Unreadable image: {path}')
            img = Image.new('RGB', default_size)
        return img

    def _load_satellite_image(self, idx):
        row_offset, col_offset = self.delta[idx, 0]
        

        sat = self._load_image(self.sat_list[self.label[idx][0]], default_size=SATELLITE_IMAGE_SIZE)
        width_raw, height_raw = sat.size[::-1]
        
        sat = self.satimage_transform(sat)

        row_offset = row_offset / height_raw * sat.size(1)
        col_offset = col_offset / width_raw * sat.size(2)
        
        return sat, row_offset, col_offset
    

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'







class VIGORArcPointDataset(Dataset):
    def __init__(self, root='/data/KHS/VIGOR', label_root=None, split=None, train=None, random_orientation=None, transform=None, first_run=None, use_filtered=False):
        self.root = root
        self.label_root = label_root
        self.split = split
        if train =='train':
            self.train=True
        else:
            self.train=False

        self.use_filtered = use_filtered
        self.random_orientation = random_orientation
        self.first_run = first_run
        self.grdimage_transform, self.satimage_transform = (transform_grd, transform_sat)

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()


        self.data_size = len(self.grd_list)

        if random_orientation:
            if not first_run:
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_pred.txt')) as file:
                    self.ori_pred = [line.rstrip() for line in file]
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_gt.txt')) as file:
                    self.ori_gt = [line.rstrip() for line in file]
    

    def _get_city_list(self):
        if self.split == 'same':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'cross':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []

    def _load_satellite_data(self):
        sat_list, sat_index_dict = [], {}
        idx = 0
        
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            # print(f'Loaded {sat_list_fname}, {idx} entries')
        
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        sat_cover_dict = {}
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)
            
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    
                    if label[0] not in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            
            # print(f'Loaded {label_fname}, {idx} entries')
        
        return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

    # def _get_label_file(self, city):
    #     if self.split == 'same':
    #         return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
    #     return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def _get_label_file(self, city): # filtered
        if self.split == 'same':
            if self.use_filtered:
                print('=== This test use filtered data ===')
                return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'Filtered_same_area_balanced_test.txt')
            else:
                return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
        else:
            if self.use_filtered:
                print('=== This test use filtered data ===')
                return os.path.join(self.root, self.label_root, city, 'Filtered_pano_label_balanced.txt')
            else:
                return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')



    def __len__(self):
        return self.data_size



    def __getitem__(self, idx):
        grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
        grd = self.grdimage_transform(grd)


        grd_points = self._load_image_depth(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE, data_type='grd')
        grd_points = torch.from_numpy(grd_points) # 3 h w


        if self.random_orientation:
            rotation = np.random.uniform(-180/360, 180/360) 
        else:
            rotation = 0

        grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
        grd_points_rolled = torch.roll(grd_points, int(round(rotation * grd_points.size(2))), dims=2)

        max_horizontal_dist = 35.5
        min_height = -10.0
        max_height = 100.0

        # 2. 각 축에 대한 마스크 생성
        # Y축(인덱스 1)을 높이로 가정합니다.
        x_mask = torch.abs(grd_points_rolled[0]) <= max_horizontal_dist
        y_mask = (grd_points_rolled[1] >= min_height) & (grd_points_rolled[1] <= max_height)
        z_mask = torch.abs(grd_points_rolled[2]) <= max_horizontal_dist

        # 3. 모든 마스크를 결합
        combined_mask = x_mask & y_mask & z_mask

        # 4. 마스크 밖의 포인트들을 NaN으로 처리
        expanded_mask = combined_mask.unsqueeze(0).expand_as(grd_points_rolled)
        grd_points_rolled[~expanded_mask] = float('nan')
        
        # 5. 좌표계 보정 (이전 코드 유지)
        grd_points_rolled[1,:,:] = -grd_points_rolled[1,:,:]
        grd_points_rolled[2,:,:] = -grd_points_rolled[2,:,:]


        yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

        sat, row_offset, col_offset = self._load_satellite_image(idx)

        gt_loc = torch.tensor([[-row_offset, col_offset]]) # gtloc2cetner translation !not center2gtloc!
        r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
        city = self._get_city_name(self.grd_list[idx])

        grd_sat_depth = [grd_points_rolled]

        return grd_rolled, sat, gt_loc, r, city, grd_sat_depth

    def _load_image(self, path, default_size):
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Unreadable image: {path}')
            img = Image.new('RGB', default_size)
        return img

    def _load_image_depth(self, path, default_size, data_type):

        if data_type == 'grd': # point cloud
            depth_resized = np.load(path.replace("panorama", "panorama_points").replace(".jpg", ".npy"), allow_pickle=True) # 3, H, W (3:coords)
        else : # sat
            img = Image.open(path).convert('I')
            depth = np.array(img, dtype=np.uint16)
            depth_tensor = torch.from_numpy(depth.astype(np.float32))[None,None,:,:] # 1,1,H,W
            depth_resized = F.interpolate(depth_tensor, size=default_size, mode='bilinear', align_corners=True).squeeze(0)

        return depth_resized # 1,H,W


    def _load_satellite_image(self, idx):
        row_offset, col_offset = self.delta[idx, 0] # pixel offset
        # row offset : y(h)
        # col offset : x(w)

        sat = self._load_image(self.sat_list[self.label[idx][0]], default_size=SATELLITE_IMAGE_SIZE)
        # sat_rel_depth = self._load_image_depth(self.sat_list[self.label[idx][0]].replace("/satellite/", "/satellite_depth/"), default_size=SATELLITE_IMAGE_SIZE, data_type='sat')
        width_raw, height_raw = sat.size[::-1]
        
        sat = self.satimage_transform(sat)

        row_offset = row_offset / height_raw * sat.size(1)
        col_offset = col_offset / width_raw * sat.size(2)
        
        return sat, row_offset, col_offset 
    

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'





class VIGORArcDataset(Dataset):
    def __init__(self, root='/data/KHS/VIGOR', label_root=None, split=None, train=None, random_orientation=None, transform=None, first_run=None, use_filtered=False):
        self.root = root
        self.label_root = label_root
        self.split = split
        if train =='train':
            self.train=True
        else:
            self.train=False

        self.use_filtered = use_filtered
        self.random_orientation = random_orientation
        self.first_run = first_run
        self.grdimage_transform, self.satimage_transform = (transform_grd, transform_sat)

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()




        # --- 1. 차원 정의 ---
        self.pano_shape = (45, 90)
        self.sat_shape = (41, 41)
        
        self.H_pano, self.W_pano = self.pano_shape
        self.H_sat, self.W_sat = self.sat_shape
        
        self.P_FLAT_SIZE = self.H_pano * self.W_pano  # 4050
        self.S_FLAT_SIZE = self.H_sat * self.W_sat  # 1681

        # --- 2. 파노라마-방향 매핑 파라미터 ---
        self.pano_center_col = self.W_pano // 2 - 1  # 44
        self.pano_end_col = self.W_pano - 1          # 89

        # --- 3. (미리 생성 1) 위성 좌표계 (41, 41) ---
        r_coords = torch.arange(self.H_sat, dtype=torch.float32)
        c_coords = torch.arange(self.W_sat, dtype=torch.float32)
        # self.sat_c_grid, self.sat_r_grid는 (41, 41) 크기를 가짐
        self.sat_c_grid, self.sat_r_grid = torch.meshgrid(c_coords, r_coords, indexing='xy')

        # --- 4. (미리 생성 2) 파노라마 인덱스 룩업 테이블 (90, 45) ---
        # k번 컬럼에 해당하는 45개의 픽셀 인덱스(i)를 미리 계산
        base_i = torch.arange(self.H_pano) * self.W_pano
        offset_k = torch.arange(self.W_pano)
        # (45, 1) + (1, 90) -> .T -> (90, 45)
        self.precomputed_i_array = (base_i.unsqueeze(1) + offset_k.unsqueeze(0)).T
        

        self.W_feat = 41
        self.H_feat = 41
        self.H_img = 656
        self.W_img = 656

        self.H_pano, self.W_pano = 45, 90
        self.P_COLS = 90
        self.ratio = self.H_img // self.H_feat
        self.pool = nn.MaxPool2d(kernel_size=self.ratio)


        r_high = torch.arange(0, self.H_img, dtype=torch.float32)
        c_high = torch.arange(0, self.W_img, dtype=torch.float32)

        self.sat_c_grid_high, self.sat_r_grid_high = torch.meshgrid(c_high, r_high, indexing='xy')
        k_i_map = torch.zeros((self.P_COLS, self.P_FLAT_SIZE), dtype=torch.float32)
        # precomputed_i_array를 인덱스로 사용하여 1을 채움
        k_i_map.scatter_(1, self.precomputed_i_array, 1.0)
        self.precomputed_k_i_map = k_i_map
        dilation_kernel_size = 5
        
        # 커널 크기에 맞는 자동 패딩 계산 (stride=1 기준)
        dilation_padding = (dilation_kernel_size - 1) // 2 # 3x3이면 1, 5x5이면 2
        
        self.dilation_pool = nn.MaxPool2d(
            kernel_size=dilation_kernel_size,
            stride=1,
            padding=dilation_padding
        )


        self.data_size = len(self.grd_list)

        if random_orientation:
            if not first_run:
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_pred.txt')) as file:
                    self.ori_pred = [line.rstrip() for line in file]
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_gt.txt')) as file:
                    self.ori_gt = [line.rstrip() for line in file]
    

    def _get_city_list(self):
        if self.split == 'same':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'cross':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []

    def _load_satellite_data(self):
        sat_list, sat_index_dict = [], {}
        idx = 0
        
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            # print(f'Loaded {sat_list_fname}, {idx} entries')
        
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        sat_cover_dict = {}
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)
            
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    
                    if label[0] not in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            
            # print(f'Loaded {label_fname}, {idx} entries')
        
        return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

    # def _get_label_file(self, city):
    #     if self.split == 'same':
    #         return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
    #     return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def _get_label_file(self, city): # filtered
        if self.split == 'same':
            if self.use_filtered:
                print('=== This test use filtered data ===')
                return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'Filtered_same_area_balanced_test.txt')
            else:
                return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
        else:
            if self.use_filtered:
                print('=== This test use filtered data ===')
                return os.path.join(self.root, self.label_root, city, 'Filtered_pano_label_balanced.txt')
            else:
                return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')



    def __len__(self):
        return self.data_size




    def __getitem__(self, idx):
        grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
        grd = self.grdimage_transform(grd)

        if self.random_orientation:
            rotation = np.random.uniform(-180/360, 180/360)
            rotation = np.random.uniform(-180/360, 180/360)
            rotation = np.random.uniform(-180/360, 180/360)
        else:
            rotation = 0

        grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
        
        yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

        sat, row_offset, col_offset = self._load_satellite_image(idx)

        gt_loc = torch.tensor([[-row_offset, col_offset]]) # gtloc2cetner translation !not center2gtloc!
        r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
        city = self._get_city_name(self.grd_list[idx])

        # # --- translation to coords ---
        # gt_coord_px = torch.zeros_like(gt_loc) 
        # center = sat.shape[-1]//2

        # gt_coord_px[:,0] = center - gt_loc[:,0] # y(h)
        # gt_coord_px[:,1] = center - gt_loc[:,1] # x(w)

        # gt_coord_bev_idx = convert_hires_to_lowres_index(gt_coord_px)
        # matching_GT = self._create_matching_gt_spatial_dilation(gt_coord_bev_idx, yaw)
        # grd_sat_depth = [matching_GT]
        grd_sat_depth = [0]

        return grd_rolled, sat, gt_loc, r, city, grd_sat_depth

    def _load_image(self, path, default_size):
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Unreadable image: {path}')
            img = Image.new('RGB', default_size)
        return img

    def _load_image_depth(self, path, default_size, data_type):

        if data_type == 'grd': # point cloud
            depth_resized = np.load(path.replace("panorama", "panorama_points").replace(".jpg", ".npy"), allow_pickle=True) # 3, H, W (3:coords)
        else : # sat
            img = Image.open(path).convert('I')
            depth = np.array(img, dtype=np.uint16)
            depth_tensor = torch.from_numpy(depth.astype(np.float32))[None,None,:,:] # 1,1,H,W
            depth_resized = F.interpolate(depth_tensor, size=default_size, mode='bilinear', align_corners=True).squeeze(0)

        return depth_resized # 1,H,W


    def _load_satellite_image(self, idx):
        row_offset, col_offset = self.delta[idx, 0] # pixel offset
        # row offset : y(h)
        # col offset : x(w)

        sat = self._load_image(self.sat_list[self.label[idx][0]], default_size=SATELLITE_IMAGE_SIZE)
        # sat_rel_depth = self._load_image_depth(self.sat_list[self.label[idx][0]].replace("/satellite/", "/satellite_depth/"), default_size=SATELLITE_IMAGE_SIZE, data_type='sat')
        width_raw, height_raw = sat.size[::-1]
        
        sat = self.satimage_transform(sat)

        row_offset = row_offset / height_raw * sat.size(1)
        col_offset = col_offset / width_raw * sat.size(2)
        
        return sat, row_offset, col_offset 
    

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'


class VIGORArcDistillDataset(Dataset):
    def __init__(self, root='/data/KHS/VIGOR', label_root=None, split=None, train=None, random_orientation=None, transform=None, first_run=None):
        self.root = root
        self.label_root = label_root
        self.split = split
        if train =='train':
            self.train=True
        else:
            self.train=False
        self.random_orientation = random_orientation
        self.first_run = first_run
        self.grdimage_transform, self.satimage_transform = (transform_grd, transform_sat)

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()



        self.data_size = len(self.grd_list)

        if random_orientation:
            if not first_run:
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_pred.txt')) as file:
                    self.ori_pred = [line.rstrip() for line in file]
                with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_gt.txt')) as file:
                    self.ori_gt = [line.rstrip() for line in file]
    

    def _get_city_list(self):
        if self.split == 'same':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'cross':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        return []

    def _load_satellite_data(self):
        sat_list, sat_index_dict = [], {}
        idx = 0
        
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            # print(f'Loaded {sat_list_fname}, {idx} entries')
        
        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        sat_cover_dict = {}
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)
            
            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    
                    if label[0] not in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            
            # print(f'Loaded {label_fname}, {idx} entries')
        
        return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

    def _get_label_file(self, city):
        if self.split == 'same':
            return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
        return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def __len__(self):
        return self.data_size





    def visualize_gt_matching_matrix(self, 
                                    M_dense_gt: torch.Tensor, 
                                    center_index_sat: int, 
                                    pano_index_to_viz: int):
        """
        Args:
            self (object): 클래스 인스턴스
            M_dense_gt (torch.Tensor): _create_matching_gt로 생성된 (S_FLAT_SIZE, P_FLAT_SIZE) 텐서
            center_index_sat (int): GT 생성 시 사용한 위성 중심의 1D 인덱스
            pano_index_to_viz (int): 시각화할 파노라마 픽셀의 1D 인덱스
        """
        
        # --- 1. 디바이스 및 타입 변환 (CPU, numpy) ---
        # GPU에 있는 GT 텐서를 CPU로 가져와 numpy로 변환
        try:
            M_dense_gt_cpu = M_dense_gt.cpu().numpy()
        except AttributeError:
            # 이미 numpy일 경우
            M_dense_gt_cpu = M_dense_gt

        
        # --- 2. 파노라마 이미지 생성 (Image1) ---
        pano_image = np.zeros((self.H_pano, self.W_pano))
        
        # 시각화할 파노라마 픽셀의 2D 좌표 계산
        pano_r = pano_index_to_viz // self.W_pano
        pano_c = pano_index_to_viz % self.W_pano
        
        # 흰색 점 찍기
        if 0 <= pano_r < self.H_pano and 0 <= pano_c < self.W_pano:
            pano_image[pano_r, pano_c] = 1.0
            
        pano_title = f"Image1 ({self.H_pano}x{self.W_pano})\nidx {pano_index_to_viz} -> (r={pano_r}, c={pano_c})"

        # --- 3. 위성 이미지 생성 (Image2) - (핵심!) ---
        
        # (3-1) M_dense_gt에서 'pano_index_to_viz' 컬럼을 통째로 가져옵니다.
        # 
        # 이 컬럼이 바로 해당 파노라마 픽셀과 매칭되는 위성 픽셀들의 값입니다.
        # shape: (S_FLAT_SIZE,)
        try:
            matching_sat_pixels_flat = M_dense_gt_cpu[:, pano_index_to_viz]
        except IndexError:
            print(f"Error: pano_index_to_viz ({pano_index_to_viz}) is out of bounds for M_dense_gt shape ({M_dense_gt_cpu.shape})")
            return
        except Exception as e:
            print(f"An error occurred during slicing: {e}")
            return
            
        # (3-2) 1D 벡터를 2D 위성 이미지 크기(e.g., 41x41)로 변환
        # shape: (H_sat, W_sat)
        sat_image = matching_sat_pixels_flat.reshape(self.H_sat, self.W_sat)
        
        # (3-3) 위성 중심점(빨간 점) 좌표 계산
        center_r_sat = center_index_sat // self.W_sat
        center_c_sat = center_index_sat % self.W_sat
        
        # 매칭된 픽셀 수 계산 (Thick GT는 1보다 클 수 있으므로 > 0)
        matched_count = np.sum(sat_image > 0) 
        sat_title = f"Image2 ({self.H_sat}x{self.W_sat})\n(Actual Matrix Slice)\nmatched: {matched_count} px"

        # --- 4. Matplotlib으로 시각화 ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot Image1 (Panorama)
        ax1.imshow(pano_image, cmap='gray', vmin=0, vmax=1)
        ax1.set_title(pano_title)
        ax1.set_xticks([])
        ax1.set_yticks([])

        # Plot Image2 (Satellite)
        # sat_image (즉, M_dense_gt의 슬라이스)를 그대로 그립니다.
        ax2.imshow(sat_image, cmap='gray', vmin=0, vmax=1)
        
        # 위성 중심에 빨간 점 오버레이
        ax2.plot(center_c_sat, center_r_sat, 'ro', markersize=8, label='Satellite Center') 
        
        ax2.set_title(sat_title)
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.show()




    def __getitem__(self, idx):
        grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
        grd = self.grdimage_transform(grd)

        if self.random_orientation:
            rotation = np.random.uniform(-180/360, 180/360) 
        else:
            rotation = 0

        grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
        
        yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

        sat, row_offset, col_offset = self._load_satellite_image(idx)

        gt_loc = torch.tensor([[-row_offset, col_offset]]) # gtloc2cetner translation !not center2gtloc!
        r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
        city = self._get_city_name(self.grd_list[idx])

        # --- translation to coords ---
        gt_coord_px = torch.zeros_like(gt_loc) 
        center = sat.shape[-1]//2

        gt_coord_px[:,0] = center - gt_loc[:,0] # y(h)
        gt_coord_px[:,1] = center - gt_loc[:,1] # x(w)

        grd_sat_depth = [0]

        return grd_rolled, grd, sat, gt_loc, r, rotation, city, grd_sat_depth

    def _load_image(self, path, default_size):
        try:
            img = Image.open(path).convert('RGB')
        except:
            print(f'Unreadable image: {path}')
            img = Image.new('RGB', default_size)
        return img

    def _load_image_depth(self, path, default_size, data_type):

        if data_type == 'grd': # point cloud
            depth_resized = np.load(path.replace("panorama", "panorama_points").replace(".jpg", ".npy"), allow_pickle=True) # 3, H, W (3:coords)
        else : # sat
            img = Image.open(path).convert('I')
            depth = np.array(img, dtype=np.uint16)
            depth_tensor = torch.from_numpy(depth.astype(np.float32))[None,None,:,:] # 1,1,H,W
            depth_resized = F.interpolate(depth_tensor, size=default_size, mode='bilinear', align_corners=True).squeeze(0)

        return depth_resized # 1,H,W


    def _load_satellite_image(self, idx):
        row_offset, col_offset = self.delta[idx, 0] # pixel offset
        # row offset : y(h)
        # col offset : x(w)

        sat = self._load_image(self.sat_list[self.label[idx][0]], default_size=SATELLITE_IMAGE_SIZE)
        # sat_rel_depth = self._load_image_depth(self.sat_list[self.label[idx][0]].replace("/satellite/", "/satellite_depth/"), default_size=SATELLITE_IMAGE_SIZE, data_type='sat')
        width_raw, height_raw = sat.size[::-1]
        
        sat = self.satimage_transform(sat)

        row_offset = row_offset / height_raw * sat.size(1)
        col_offset = col_offset / width_raw * sat.size(2)
        
        return sat, row_offset, col_offset 
    

    def _get_city_name(self, path):
        for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
            if city in path:
                return city
        return 'Unknown'


# class VIGORArcPointDataset(Dataset):
#     def __init__(self, root='/data/KHS/VIGOR', label_root=None, split=None, train=None, random_orientation=None, transform=None, first_run=None):
#         self.root = root
#         self.label_root = label_root
#         self.split = split
#         if train =='train':
#             self.train=True
#         else:
#             self.train=False
#         self.random_orientation = random_orientation
#         self.first_run = first_run
#         self.grdimage_transform, self.satimage_transform = (transform_grd, transform_sat)

#         self.city_list = self._get_city_list()
#         self.sat_list, self.sat_index_dict = self._load_satellite_data()
#         self.grd_list, self.label, self.delta, self.sat_cover_dict = self._load_ground_data()




#         # --- 1. 차원 정의 ---
#         self.pano_shape = (45, 90)
#         self.sat_shape = (41, 41)
        
#         self.H_pano, self.W_pano = self.pano_shape
#         self.H_sat, self.W_sat = self.sat_shape
        
#         self.P_FLAT_SIZE = self.H_pano * self.W_pano  # 4050
#         self.S_FLAT_SIZE = self.H_sat * self.W_sat  # 1681

#         # --- 2. 파노라마-방향 매핑 파라미터 ---
#         self.pano_center_col = self.W_pano // 2 - 1  # 44
#         self.pano_end_col = self.W_pano - 1          # 89

#         # --- 3. (미리 생성 1) 위성 좌표계 (41, 41) ---
#         r_coords = torch.arange(self.H_sat, dtype=torch.float32)
#         c_coords = torch.arange(self.W_sat, dtype=torch.float32)
#         # self.sat_c_grid, self.sat_r_grid는 (41, 41) 크기를 가짐
#         self.sat_c_grid, self.sat_r_grid = torch.meshgrid(c_coords, r_coords, indexing='xy')

#         # --- 4. (미리 생성 2) 파노라마 인덱스 룩업 테이블 (90, 45) ---
#         # k번 컬럼에 해당하는 45개의 픽셀 인덱스(i)를 미리 계산
#         base_i = torch.arange(self.H_pano) * self.W_pano
#         offset_k = torch.arange(self.W_pano)
#         # (45, 1) + (1, 90) -> .T -> (90, 45)
#         self.precomputed_i_array = (base_i.unsqueeze(1) + offset_k.unsqueeze(0)).T
        





#         self.data_size = len(self.grd_list)

#         if random_orientation:
#             if not first_run:
#                 with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_pred.txt')) as file:
#                     self.ori_pred = [line.rstrip() for line in file]
#                 with open(os.path.join('results', 'vigor', split,'unknown_ori', 'first_run', 'ori_gt.txt')) as file:
#                     self.ori_gt = [line.rstrip() for line in file]
    

#     def _get_city_list(self):
#         if self.split == 'same':
#             return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
#         if self.split == 'cross':
#             return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
#         return []

#     def _load_satellite_data(self):
#         sat_list, sat_index_dict = [], {}
#         idx = 0
        
#         for city in self.city_list:
#             sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
#             with open(sat_list_fname, 'r') as file:
#                 for line in file:
#                     sat_path = os.path.join(self.root, city, 'satellite', line.strip())
#                     sat_list.append(sat_path)
#                     sat_index_dict[line.strip()] = idx
#                     idx += 1
#             # print(f'Loaded {sat_list_fname}, {idx} entries')
        
#         return np.array(sat_list), sat_index_dict

#     def _load_ground_data(self):
#         grd_list, label_list, delta_list = [], [], []
#         sat_cover_dict = {}
#         idx = 0

#         for city in self.city_list:
#             label_fname = self._get_label_file(city)
            
#             with open(label_fname, 'r') as file:
#                 for line in file:
#                     data = np.array(line.split())
#                     label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
#                     delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)
                    
#                     grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
#                     label_list.append(label)
#                     delta_list.append(delta)
                    
#                     if label[0] not in sat_cover_dict:
#                         sat_cover_dict[label[0]] = [idx]
#                     else:
#                         sat_cover_dict[label[0]].append(idx)
#                     idx += 1
            
#             # print(f'Loaded {label_fname}, {idx} entries')
        
#         return grd_list, np.array(label_list), np.array(delta_list), sat_cover_dict

#     def _get_label_file(self, city):
#         if self.split == 'same':
#             return os.path.join(self.root, self.label_root, city, 'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt')
#         return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

#     def __len__(self):
#         return self.data_size


#     def _create_matching_gt(self, center_index):
#         """
#         [PyTorch 버전 클래스 메서드]
#         __init__에서 미리 계산된 텐서들을 사용하여
#         (S_FLAT_SIZE, P_FLAT_SIZE) 크기의 Dense GT 텐서를 생성합니다.
        
#         Args:
#             center_index (int): 위성 그리드의 1D flatten() 기준 인덱스 (0 ~ 1680)
            
#         Returns:
#             torch.Tensor: (1681, 4050) 크기의 Dense GT 텐서
#         """
        
#         # --- 1. 기준 인덱스 -> 2D 변환 ---
#         center_r = center_index // self.W_sat
#         center_c = center_index % self.W_sat

#         # --- 2. 위성 픽셀별 각도 계산 (미리 생성한 그리드 사용) ---
#         dy = self.sat_r_grid - center_r
#         dx = self.sat_c_grid - center_c
#         angle_rad = torch.atan2(dx, -dy)

#         # --- 3. 위성 픽셀(j) -> 파노라마 컬럼(k) 매핑 맵 (Torch) ---
#         pano_indices_map = torch.zeros((self.H_sat, self.W_sat), dtype=torch.int64)
        
#         mask_right = (angle_rad >= 0)
#         frac_right = angle_rad[mask_right] / torch.pi
#         pano_indices_map[mask_right] = torch.round(
#             self.pano_center_col + frac_right * (self.pano_end_col - self.pano_center_col)
#         ).long()

#         mask_left = (angle_rad < 0)
#         frac_left = angle_rad[mask_left] / -torch.pi
#         pano_indices_map[mask_left] = torch.round(
#             self.pano_center_col - frac_left * (self.pano_center_col - 0)
#         ).long()
        
#         pano_indices_map = torch.clip(pano_indices_map, 0, self.pano_end_col)

#         # --- 4. 희소 텐서 인덱스 생성 (미리 생성한 룩업 테이블 사용) ---
#         j_indices = torch.arange(self.S_FLAT_SIZE)
#         cols_j = torch.repeat_interleave(j_indices, self.H_pano)
        
#         # 룩업 테이블(self.precomputed_i_array)에서 파노라마 인덱스(i) 조회
#         pano_indices_flat = pano_indices_map.flatten()
#         rows_i = self.precomputed_i_array[pano_indices_flat].flatten()

#         # --- 5. 희소 텐서 생성 및 Dense 텐서로 변환 ---
#         indices = torch.stack([cols_j, rows_i], dim=0)
#         values = torch.ones_like(rows_i, dtype=torch.float32)
        
#         M_sparse_torch = torch.sparse_coo_tensor(
#             indices=indices,
#             values=values,
#             size=(self.S_FLAT_SIZE, self.P_FLAT_SIZE)
#         )
        
#         # 모델 출력(Dense)과 비교하기 위해 Dense 텐서로 변환
#         M_dense_gt = M_sparse_torch.to_dense()
        
#         return M_dense_gt
    


#     def __getitem__(self, idx):
#         grd = self._load_image(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE)
#         grd = self.grdimage_transform(grd)

#         grd_points = self._load_image_depth(self.grd_list[idx], default_size=GROUND_IMAGE_SIZE, data_type='grd')
#         grd_points = torch.from_numpy(grd_points) # 3 h w

#         if self.random_orientation:
#             rotation = np.random.uniform(-180/360, 180/360) 
#         else:
#             rotation = 0

#         grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
#         grd_points_rolled = torch.roll(grd_points, int(round(rotation * grd_points.size(2))), dims=2)
        
#         max_horizontal_dist = 35.5
#         min_height = -10.0
#         max_height = 100.0


#         x_mask = torch.abs(grd_points_rolled[0]) <= max_horizontal_dist
#         y_mask = (grd_points_rolled[1] >= min_height) & (grd_points_rolled[1] <= max_height)
#         z_mask = torch.abs(grd_points_rolled[2]) <= max_horizontal_dist

#         # 3. 모든 마스크를 결합
#         combined_mask = x_mask & y_mask & z_mask

#         # 4. 마스크 밖의 포인트들을 NaN으로 처리
#         expanded_mask = combined_mask.unsqueeze(0).expand_as(grd_points_rolled)
#         grd_points_rolled[~expanded_mask] = float('nan')
        
#         # 5. 좌표계 보정 (이전 코드 유지)
#         grd_points_rolled[1,:,:] = -grd_points_rolled[1,:,:]
#         grd_points_rolled[2,:,:] = -grd_points_rolled[2,:,:]

#         yaw = -rotation * 360 * (np.pi/180)  # 0 means heading North, clockwise increasing

#         sat, sat_rel_depth, row_offset, col_offset = self._load_satellite_image(idx)

#         gt_loc = torch.tensor([[-row_offset, col_offset]]) # gtloc2cetner translation !not center2gtloc!
#         r = torch.tensor([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]).to(torch.float32)
        
#         city = self._get_city_name(self.grd_list[idx])
#         # grd_sat_depth = [grd_points_rolled, sat_rel_depth]

#         # --- translation to coords ---
#         gt_coord_px = torch.zeros_like(gt_loc) 
#         center = sat.shape[-1]//2

#         gt_coord_px[:,0] = center - gt_loc[:,0] # y(h)
#         gt_coord_px[:,1] = center - gt_loc[:,1] # x(w)

#         gt_coord_bev_idx = convert_hires_to_lowres_index(gt_coord_px)
#         matching_GT = self._create_matching_gt(gt_coord_bev_idx)
#         grd_sat_depth = [grd_points_rolled, sat_rel_depth, matching_GT]

#         return grd_rolled, sat, gt_loc, r, city, grd_sat_depth

#     def _load_image(self, path, default_size):
#         try:
#             img = Image.open(path).convert('RGB')
#         except:
#             print(f'Unreadable image: {path}')
#             img = Image.new('RGB', default_size)
#         return img

#     def _load_image_depth(self, path, default_size, data_type):

#         if data_type == 'grd': # point cloud
#             depth_resized = np.load(path.replace("panorama", "panorama_points").replace(".jpg", ".npy"), allow_pickle=True) # 3, H, W (3:coords)
#         else : # sat
#             img = Image.open(path).convert('I')
#             depth = np.array(img, dtype=np.uint16)
#             depth_tensor = torch.from_numpy(depth.astype(np.float32))[None,None,:,:] # 1,1,H,W
#             depth_resized = F.interpolate(depth_tensor, size=default_size, mode='bilinear', align_corners=True).squeeze(0)

#         return depth_resized # 1,H,W


#     def _load_satellite_image(self, idx):
#         row_offset, col_offset = self.delta[idx, 0] # pixel offset
#         # row offset : y(h)
#         # col offset : x(w)

#         sat = self._load_image(self.sat_list[self.label[idx][0]], default_size=SATELLITE_IMAGE_SIZE)
#         sat_rel_depth = self._load_image_depth(self.sat_list[self.label[idx][0]].replace("/satellite/", "/satellite_depth/"), default_size=SATELLITE_IMAGE_SIZE, data_type='sat')
#         width_raw, height_raw = sat.size[::-1]
        
#         sat = self.satimage_transform(sat)

#         row_offset = row_offset / height_raw * sat.size(1)
#         col_offset = col_offset / width_raw * sat.size(2)
        
#         return sat, sat_rel_depth, row_offset, col_offset 
    

#     def _get_city_name(self, path):
#         for city in ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']:
#             if city in path:
#                 return city
#         return 'Unknown'




def convert_hires_to_lowres_index(coords_px, image_shape=(640,640), bev_grid_shape=(41,41)):
    """
    Args:
        coords_px (torch.tensor): [x, y] pixel level coords.
        image_shape (tuple): (H, W) img size (640, 640).
        bev_grid_shape (tuple): (H, W) bev grid size  ex.(41, 41).

    Returns:
        int: bev flatten() idx
    """
    
    # 스케일링 팩터
    bev_W = bev_grid_shape[0]
    bev_H = bev_grid_shape[1]
    img_W = image_shape[0]
    img_H = image_shape[0]

    scale_factor_x = bev_W / img_W
    scale_factor_y = bev_H / img_H
    
    # Low-Res 타겟 좌표 (float x, float y)
    target_bev_x = coords_px[:,0] * scale_factor_x
    target_bev_y = coords_px[:,1] * scale_factor_y

    # x 좌표 -> col 인덱스
    # y 좌표 -> row 인덱스
    col_int = int(torch.floor(target_bev_x))
    row_int = int(torch.floor(target_bev_y))

    # col_int = torch.clip(col_int, 0, bev_W - 1)
    # row_int = torch.clip(row_int, 0, bev_H - 1)

    # (row * 너비) + col
    flatten_index = (row_int * bev_W) + col_int
    
    return flatten_index
