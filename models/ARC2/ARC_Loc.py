import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import DeepResBlock_desc, DeepResBlock_kpts

import random
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")
seed = config.getint("RandomSeed", "seed")
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(mode=True, warn_only=True)

class CVM(nn.Module):
    def __init__(self, device, grd_enc, sat_enc, grd_bev_res, sat_bev_res, grid_size_h, grid_size_v,
                temperature=0.1, embed_dim=1024, desc_dim=128):
    # def __init__(self, device, grd_enc, grd_bev_res, sat_bev_res, grid_size_h, grid_size_v,
    #             temperature=0.1, embed_dim=1024, desc_dim=128):
        super(CVM, self).__init__()

        device = f'cuda:{device}'
        self.grd_encoder = grd_enc
        self.sat_encoder = sat_enc

        self.temperature = temperature
        self.embed_dim = embed_dim

        self.grd_bev_res = grd_bev_res
        self.sat_bev_res = sat_bev_res

        bn = True
        block_dims = [512, 256, 128, 128]
        add_posEnc = True
        norm_desc = True

        
        
        self.dustbin_score = nn.Parameter(torch.tensor(1.))

        self.grd_projector =  DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
        self.sat_projector =  DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
    
        self.grd_conf_head = DeepResBlock_kpts(bn=bn, in_channels=self.embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, use_softmax=None)
        self.sat_conf_head = DeepResBlock_kpts(bn=bn, in_channels=self.embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, use_softmax=None)


    def forward(self, grd, sat, other=None):

        ## =============== Feature Extraction ===============
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                grd_feature = self.grd_encoder(grd, feature_fmt='NCHW').features # RADIO feature
                sat_feature = self.sat_encoder(sat, feature_fmt='NCHW').features # RADIO feature



        ## =============== Ground ===============
        bs = grd_feature.size()[0]

        grd_desc = self.grd_projector(grd_feature).flatten(2) # (B, C_desc, H_g, W_g)        
        grd_conf_map = self.grd_conf_head(grd_feature).flatten(2).squeeze(1) # (B, 1, H_g, W_g)



        ## =============== Satellite ===============
        sat_desc = self.sat_projector(sat_feature).flatten(2) # (B, C_desc, M)
        sat_conf_map = self.sat_conf_head(sat_feature).flatten(2).squeeze(1) # (B, 1, H_s, W_s)



        ## =============== Matching score ===============   
        matching_score_original = torch.matmul(sat_desc.transpose(1, 2).contiguous(), grd_desc) / self.temperature
        
        b, m, n = matching_score_original.shape

        bins0 = self.dustbin_score.expand(b, m, 1)
        bins1 = self.dustbin_score.expand(b, 1, n)
        alpha = self.dustbin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                                torch.cat([bins1, alpha], -1)], 1)

        match_dist_with_bins = F.softmax(couplings, 1) * F.softmax(couplings, 2)
        match_dist = match_dist_with_bins[:, :-1, :-1] # (B, M, N)

        keypoint_dist = sat_conf_map.unsqueeze(2) * grd_conf_map.unsqueeze(1) # (B, M, N)

        final_correspondence_prob = match_dist * keypoint_dist

        return final_correspondence_prob, matching_score_original, 0


from sklearn.decomposition import PCA
import torchvision.transforms.functional as TF
import os
from PIL import Image

def save_sat_desc_pca(sat_desc, save_dir='vis/sat_feat/baseline', H=41, W=41):
    os.makedirs(save_dir, exist_ok=True)
    
    B, C, N = sat_desc.shape
    sat_desc_np = sat_desc.detach().cpu().numpy()  # [B, C, N]
    
    for i in range(B):
        desc = sat_desc_np[i]  # [C, N]
        desc_T = desc.T  # [N, C]
        
        # PCA: C -> 3
        pca = PCA(n_components=3)
        desc_pca = pca.fit_transform(desc_T)  # [N, 3]
        
        # Normalize to [0, 255]
        desc_pca -= desc_pca.min()
        desc_pca /= desc_pca.max()
        desc_pca *= 255.0
        desc_pca = desc_pca.astype(np.uint8)
        
        # Reshape to [H, W, 3]
        img = desc_pca.reshape(H, W, 3)
        
        # Save
        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(save_dir, f"points_vis_{i}.png"))

