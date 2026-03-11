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
        super(CVM, self).__init__()

        device = f'cuda:{device}'
        self.grd_encoder = grd_enc
        self.sat_encoder = sat_enc

        self.sat_bev_res = sat_bev_res
        self.grd_bev_res = grd_bev_res

        self.temperature = temperature
        self.embed_dim = embed_dim 
        
        self.grd_h, self.grd_w = self.grd_bev_res[0], self.grd_bev_res[1] 
        self.pos_enc_grd = PositionalEncoding2D(embed_dim, self.grd_h, self.grd_w)

        # Satellite Feature Map Size (H=41, W=41)
        self.sat_h, self.sat_w = self.sat_bev_res, self.sat_bev_res
        self.pos_enc_sat = PositionalEncoding2D(embed_dim, self.sat_h, self.sat_w)

        # Context Module (사이즈 무관하게 작동함)
        self.context_module = ContextAwareModule(d_model=embed_dim, nhead=4)
        
        # =====================================================================

        
        
        bn = True
        block_dims = [512, 256, 128, 128]
        add_posEnc = True
        norm_desc = True
        
        self.dustbin_score = nn.Parameter(torch.tensor(1.))

        self.grd_projector = DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
        self.sat_projector = DeepResBlock_desc(bn, last_dim=desc_dim, in_channels=embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, norm_desc=norm_desc)
    
        self.grd_conf_head = DeepResBlock_kpts(bn=bn, in_channels=self.embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, use_softmax=True)
        self.sat_conf_head = DeepResBlock_kpts(bn=bn, in_channels=self.embed_dim, block_dims=block_dims, add_posEnc=add_posEnc, use_softmax=True)

    def forward(self, grd, sat, other=None):

        ## =============== Feature Extraction ===============
        with torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                grd_feature = self.grd_encoder(grd, feature_fmt='NCHW').features 
                sat_feature = self.sat_encoder(sat, feature_fmt='NCHW').features 
                
                # [안전장치] 만약 입력 사이즈가 달라질 수 있다면 여기서 Interpolation 해도 되지만,
                # 보통 고정 사이즈이므로 그대로 진행합니다.
                # print(grd_feature.shape) -> [B, 1024, 45, 90] 확인
                # print(sat_feature.shape) -> [B, 1024, 41, 41] 확인

        ## =============== Contextual Awareness ===============
        # 1. Positional Encoding Add
        # (만약 입력 크기가 dynamic하다면 forward 안에서 PE를 매번 생성해야 하지만, 고정이면 이게 빠름)
        grd_feature_pe = self.pos_enc_grd(grd_feature)
        sat_feature_pe = self.pos_enc_sat(sat_feature)

        # 2. Attention Interaction (Cross Attention handles different shapes automatically)
        # grd_feature: [B, 1024, 45, 90] -> (flatten) -> [B, 4050, 1024]
        # sat_feature: [B, 1024, 41, 41] -> (flatten) -> [B, 1681, 1024]
        # 서로 Q, K, V 교환하며 연산 후 원래 shape으로 복원되어 나옴
        grd_feature, sat_feature = self.context_module(grd_feature_pe, sat_feature_pe)
        
        ## =============== Ground Head ===============
        grd_desc = self.grd_projector(grd_feature).flatten(2)         
        grd_conf_map = self.grd_conf_head(grd_feature).flatten(2).squeeze(1) 

        ## =============== Satellite Head ===============
        sat_desc = self.sat_projector(sat_feature).flatten(2) 
        sat_conf_map = self.sat_conf_head(sat_feature).flatten(2).squeeze(1) 

        ## =============== Matching ===============   
        # [C, 4050] x [C, 1681]^T -> [4050, 1681] 매칭 매트릭스 생성됨
        matching_score_original = torch.matmul(sat_desc.transpose(1, 2).contiguous(), grd_desc) / self.temperature
        
        # ... (이하 로직 동일) ...
        b, m, n = matching_score_original.shape # m=1681, n=4050

        bins0 = self.dustbin_score.expand(b, m, 1)
        bins1 = self.dustbin_score.expand(b, 1, n)
        alpha = self.dustbin_score.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([matching_score_original, bins0], -1),
                                torch.cat([bins1, alpha], -1)], 1)

        match_dist_with_bins = F.softmax(couplings, 1) * F.softmax(couplings, 2)
        match_dist = match_dist_with_bins[:, :-1, :-1] 

        keypoint_dist = sat_conf_map.unsqueeze(2) * grd_conf_map.unsqueeze(1) 

        final_correspondence_prob = match_dist * keypoint_dist

        return final_correspondence_prob, matching_score_original, 0


import math

# ... (이전 import 및 seed 설정 코드는 동일) ...

# 1. 2D Positional Encoding (Transformer에는 필수)
class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height, width):
        super(PositionalEncoding2D, self).__init__()
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, C, H, W)
        return x + self.pe[:, :x.size(2), :x.size(3)]

# 2. Contextual Awareness Module (Self + Cross Attention)
class ContextAwareModule(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        # Self-Attention Layers
        self.self_attn_grd = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn_sat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Cross-Attention Layers (Interaction)
        self.cross_attn_grd = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn_sat = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Feed Forward Networks
        self.ffn_grd = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )
        self.ffn_sat = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        # Norms
        self.norm1_grd = nn.LayerNorm(d_model)
        self.norm2_grd = nn.LayerNorm(d_model)
        self.norm3_grd = nn.LayerNorm(d_model)

        self.norm1_sat = nn.LayerNorm(d_model)
        self.norm2_sat = nn.LayerNorm(d_model)
        self.norm3_sat = nn.LayerNorm(d_model)

    def forward(self, grd_feat, sat_feat):
        """
        Input: (B, C, H, W)
        Output: (B, C, H, W) -> Enhanced Features
        """
        B, C, H_g, W_g = grd_feat.shape
        _, _, H_s, W_s = sat_feat.shape

        # Flatten spatial dimensions: (B, C, H, W) -> (B, H*W, C)
        grd_flat = grd_feat.flatten(2).transpose(1, 2)
        sat_flat = sat_feat.flatten(2).transpose(1, 2)

        # 1. Self-Attention (Context Aggregation)
        # Ground
        grd_src = grd_flat
        grd_src2 = self.self_attn_grd(grd_src, grd_src, grd_src)[0]
        grd_src = self.norm1_grd(grd_src + grd_src2)
        
        # Satellite
        sat_src = sat_flat
        sat_src2 = self.self_attn_sat(sat_src, sat_src, sat_src)[0]
        sat_src = self.norm1_sat(sat_src + sat_src2)

        # 2. Cross-Attention (Mutual Interaction) - 핵심 부분
        # Grd looks at Sat
        grd_src2 = self.cross_attn_grd(grd_src, sat_src, sat_src)[0]
        grd_src = self.norm2_grd(grd_src + grd_src2)

        # Sat looks at Grd
        sat_src2 = self.cross_attn_sat(sat_src, grd_src, grd_src)[0]
        sat_src = self.norm2_sat(sat_src + sat_src2)

        # 3. FFN
        grd_src = self.norm3_grd(grd_src + self.ffn_grd(grd_src))
        sat_src = self.norm3_sat(sat_src + self.ffn_sat(sat_src))

        # Reshape back to (B, C, H, W)
        grd_out = grd_src.transpose(1, 2).reshape(B, C, H_g, W_g)
        sat_out = sat_src.transpose(1, 2).reshape(B, C, H_s, W_s)

        return grd_out, sat_out

