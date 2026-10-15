import torch
import torch.nn.functional as F
import numpy as np
import math
import os

def simultaneous_sampling(weights: torch.Tensor, K: int, Hnum: int) -> torch.Tensor:
    """
    weights: (B, Msets, N) >=0
    return:  indices (B, Msets, Hnum, K)
    """
    eps = 1e-20
    w = weights.clamp_min(eps)
    logw = w.log()  # (B,M,N)
    B, M, N = logw.shape
    U = torch.rand(B, M, Hnum, N, device=w.device, dtype=w.dtype).clamp_min(eps)
    G = -(-U.log()).log()  # Gumbel(0,1)
    scores = logw.unsqueeze(2) + G  # (B,M,H,N)
    return scores.topk(k=K, dim=-1).indices  # (B,M,H,K)

class ARCSolverRANSAC:
    """
    - e2eBearingLSSolverRANSAC과 동일 인터페이스/출력.
    - it_RANSAC: 가설 수(H) (병렬 처리, 100)
    - it_matches: 큰 셋 개수(M) (각각 N개 라인으로 구성, 20)
    - num_samples_matches: 큰 셋 크기(N, 8192)
    - num_corr_lines: 최소 샘플셋 크기(K, 최소 2)
    """
    def __init__(self, it_RANSAC, it_matches, num_samples_matches, num_corr_lines,
                 num_refinements, th_inlier, th_soft_inlier,
                 Hs, Ws, grd_bev_res, sat_bev_res):
        self.H = it_RANSAC
        self.M = it_matches
        self.N = num_samples_matches
        self.K = num_corr_lines
        self.R = num_refinements
        self.th_inlier = float(th_inlier)
        self.th_soft = float(th_soft_inlier)

        self.Hs = Hs
        self.Ws = Ws
        self.grd_bev_res = grd_bev_res
        self.sat_bev_res = sat_bev_res

    @staticmethod
    # def _lines_from_pixels_and_dirs(sat_x, sat_y, u, rgt=None):
    #     # u[...,0]=ux(+East), u[...,1]=uy(+North)
    #     u = u @ rgt # 방향보정
    #     a = u[..., 1]  # uy
    #     b = u[..., 0]  # ux
    #     c = -(a * sat_x + b * sat_y)
    #     return torch.stack([a, b, c], dim=-1)  # (...,N,3)

    def _lines_from_pixels_and_dirs(sat_x, sat_y, u):
        # u[...,0]=ux(+East), u[...,1]=uy(+North)
        a = u[..., 1]  # uy
        b = u[..., 0]  # ux
        c = -(a * sat_x + b * sat_y)
        return torch.stack([a, b, c], dim=-1)  # (...,N,3)

    @staticmethod
    def _dist_point_to_lines(p, lines):
        # p: (...,2), lines: (...,N,3)  -> (...,N)
        px = p[..., 0].unsqueeze(-1)
        py = p[..., 1].unsqueeze(-1)
        a = lines[..., 0]; b = lines[..., 1]; c = lines[..., 2]
        return (a * px + b * py + c).abs()  # a^2+b^2≈1 가정

    def _soft_score(self, lines, p, weights, th_soft):
        # lines: (B,N,3), p: (B,*,2)  -> score: (B,*) ; * = H
        dist = self._dist_point_to_lines(p.unsqueeze(-2), lines.unsqueeze(-3))  # (B,*,N)
        score = weights.unsqueeze(-2) / (1.0 + (dist / max(th_soft, 1e-9))**2)
        # score = 1 / (1.0 + (dist / max(th_soft, 1e-9))**2) # 테스트용 스코어 코드 한줄
        return score.sum(dim=-1)  # (B,*)

    def estimate_pose(self, matching_score: torch.Tensor, matching_score_orig: torch.Tensor, return_inliers: bool = False):
        """
        반환:
          R_dummy: (B,2,2) identity
          pstar_best: (B,2)  (x,y) in pixels
          best_inliers: (B,) soft score
          inliers_list: None
        """
        device = matching_score.device
        dtype  = matching_score.dtype
        B, Ms, Ng = matching_score.shape
        # pano width 계산 규약 동일 유지
        pano_width = 2 * int((Ng // 2) ** 0.5)

        # ---------- 1) M개 큰 셋(N 라인)을 배치로 구성 ----------
        weights_all = matching_score.reshape(B, Ms * Ng).clamp_min(1e-12)  # (B, Ms*Ng)
        # 각 match-set마다 N개 샘플링 → (B,M,N)
        sampled_idx = torch.stack(
            [torch.multinomial(weights_all, self.N, replacement=False) for _ in range(self.M)],
            dim=1
        )  # (B,M,N)

        # sat/grd 분해
        sampled_idx_sat = torch.div(sampled_idx, Ng, rounding_mode='trunc')  # (B,M,N)
        sampled_idx_grd = sampled_idx % Ng                                   # (B,M,N)
        # 큰 셋 가중치
        weights_large = torch.gather(weights_all, 1, sampled_idx.view(B, -1)).view(B, self.M, self.N)  # (B,M,N)

        # 좌표/방위
        sat_y, sat_x = grid_idx_to_sat_pixels(sampled_idx_sat, self.sat_bev_res, self.Hs, self.Ws)   # (B,M,N)
        grd_x_idx = sampled_idx_grd % pano_width
        u_large = pano_dirs_from_indices(grd_x_idx, pano_width)  # (B,M,N,2)

        # 직선 (B,M,N,3)
        lines_large = self._lines_from_pixels_and_dirs(sat_x, sat_y, u_large)
        # lines_large = self._lines_from_pixels_and_dirs(sat_x, sat_y, u_large)

        # ---------- 2) 각 큰 셋에서 H개 가설의 최소셋(K) 동시 샘플 ----------
        idx_k = simultaneous_sampling(weights_large, self.K, self.H)  # (B,M,H,K)

        b_idx = torch.arange(B, device=device)[:, None, None, None].expand(B, self.M, self.H, self.K)
        m_idx = torch.arange(self.M, device=device)[None, :, None, None].expand(B, self.M, self.H, self.K)

        lines_k   = lines_large[b_idx, m_idx, idx_k, :]     # (B,M,H,K,3)
        weights_k = weights_large[b_idx, m_idx, idx_k]      # (B,M,H,K)

        # ---------- 3) 모든 가설 일괄 LS 해 (B,M,H,2) ----------
        a = lines_k[..., 0]; b = lines_k[..., 1]; c = lines_k[..., 2]; w = weights_k
        M11 = (w * a * a).sum(dim=-1)
        M12 = (w * a * b).sum(dim=-1)
        M22 = (w * b * b).sum(dim=-1)
        r1  = -(w * a * c).sum(dim=-1)
        r2  = -(w * b * c).sum(dim=-1)
        Mmat = torch.stack([torch.stack([M11, M12], dim=-1),
                            torch.stack([M12, M22], dim=-1)], dim=-2)    # (B,M,H,2,2)
        rhs  = torch.stack([r1, r2], dim=-1).unsqueeze(-1)               # (B,M,H,2,1)
        # pstar_k = torch.linalg.solve(Mmat, rhs).squeeze(-1)              # (B,M,H,2)

        mean_diag = 0.5 * (M11 + M22)                          # (B,M,H)
        lam = (1e-3 * mean_diag).clamp_min(1e-9)                  # (B,M,H)
        I2 = torch.eye(2, device=Mmat.device, dtype=Mmat.dtype).view(1,1,1,2,2)
        Mmat_damped = Mmat + lam.unsqueeze(-1).unsqueeze(-1) * I2  # (B,M,H,2,2)
        Mpinv = torch.linalg.pinv(Mmat_damped, rcond=1e-6)               # (B,M,H,2,2)
        pstar_k = (Mpinv @ rhs).squeeze(-1)                    # (B,M,H,2)

        sum_w = w.sum(dim=-1)
        invalid = (sum_w < 1e-8)
        svals = torch.linalg.svdvals(Mmat_damped)
        smin = svals[..., -1]
        invalid = invalid | (smin < 1e-8)

        # ---------- 4) 각 가설을 자기 큰 셋(N)에 대해 일괄 스코어 ----------
        # dist: (B,M,H,N)
        px = pstar_k[..., 0].unsqueeze(-1)
        py = pstar_k[..., 1].unsqueeze(-1)
        aL = lines_large[..., 0].unsqueeze(2)
        bL = lines_large[..., 1].unsqueeze(2)
        cL = lines_large[..., 2].unsqueeze(2)
        dist = (aL * px + bL * py + cL).abs()
        score = (weights_large.unsqueeze(2) / (1.0 + (dist / max(self.th_soft, 1e-9))**2)).sum(dim=-1)  # (B,M,H)

        # NaN/Inf 가설은 큰 음수로 마스킹
        invalid = torch.isnan(pstar_k).any(dim=-1) | torch.isinf(pstar_k).any(dim=-1)  # (B,M,H)
        score = score.masked_fill(invalid, -1e9)

        # ---------- 5) 배치별 최고 가설 선택 ----------
        score_flat = score.view(B, -1)                  # (B, M*H)
        max_ind = torch.argmax(score_flat, dim=1)       # (B,)
        m_best = (max_ind // self.H)                    # (B,)
        h_best = (max_ind %  self.H)                    # (B,)

        p_best = pstar_k[torch.arange(B, device=device), m_best, h_best]   # (B,2)
        lines_best   = lines_large[torch.arange(B, device=device), m_best] # (B,N,3)
        weights_best = weights_large[torch.arange(B, device=device), m_best]# (B,N)

        # ---------- 6) 벡터라이즈 리핏 ----------
        def _soft_score_single(p, lines, w, th):
            d = self._dist_point_to_lines(p, lines)
            return (w / (1.0 + (d / max(th, 1e-9))**2)).sum(dim=-1)

        # ----- Refinement -----
        best_inliers = _soft_score_single(p_best, lines_best, weights_best, self.th_inlier)
        inliers_prev = torch.full((B,), float(self.K), device=device, dtype=weights_best.dtype)

        for _ in range(self.R):
            d_ref = self._dist_point_to_lines(p_best, lines_best)   # (B,N)
            in_mask = (d_ref < self.th_inlier)                      # (B,N)
            cnt = in_mask.sum(dim=-1).float()                       # (B,)

            do_ref = (cnt >= self.K) & (cnt > inliers_prev)         # (B,)
            if do_ref.sum() == 0:
                break
            inliers_prev = torch.where(do_ref, cnt, inliers_prev)

            # 부분집합 인덱스 추출
            idx = torch.nonzero(do_ref, as_tuple=False).squeeze(1)  # (B_sub,)
            # 가중치 재계산
            w_ref = (in_mask.float() * weights_best).clamp_min(1e-9)
            # 부분집합에 대해만 재추정
            p_ref = least_squares_point_to_lines(lines_best[idx], w_ref[idx])  # (B_sub, 2)
            # 결과를 원본 배치에 반영 (인덱스 할당)
            p_best[idx] = p_ref

            # 점수도 부분집합에 대해서만 갱신
            best_inliers[idx] = _soft_score_single(p_best[idx], lines_best[idx], weights_best[idx], self.th_inlier)


        # ---------- 7) 최종 인라이어 마스크 & 인덱스 ---------- 
        d_final = self._dist_point_to_lines(p_best, lines_best)   # (B,N)
        in_mask_final = (d_final < self.th_inlier)                # (B,N)

        inliers_list: list[torch.Tensor] = []
        for b in range(B):
            inlier_idx_b = torch.nonzero(in_mask_final[b], as_tuple=False).squeeze(1)  # (Nb_inlier,)
            inliers_list.append(inlier_idx_b)


        # ---------- 출력 ----------
        R_dummy = torch.eye(2, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
        # inliers_list = None
        return R_dummy, p_best, best_inliers, inliers_list






class e2eUnknownOri_KITTI:
    """
    KITTI unknown-orientation solver without RANSAC.

    Returns:
      yaw_best: (B,) rad
      p_best:   (B,2) in pixels (x,y)
      score:    (B,) soft score
    """

    def __init__(
        self,
        # yaw search
        yaw_deg_step: float = 1.0,
        yaw_full360: bool = True,
        yaw_candidates_deg: list[float] | None = None,
        # interface compatibility (unused but accepted)
        H_yaw: int = 24,
        M_yaw: int = 6,
        N_yaw: int = 2048,
        it_RANSAC: int = 100,
        it_matches: int = 20,
        num_samples_matches: int = 1024,
        num_corr_lines: int = 2,
        num_refinements: int = 3,
        # geometry / scoring
        th_inlier: float = 35.0,
        th_soft_inlier: float = 70.0,
        Hs: int = 656,
        Ws: int = 656,
        grd_img_size_x: int = 1232,
        grd_bev_res: int = 41,
        sat_bev_res: int = 41,
        disamb_beta: float = 6.0,
        disamb_scale: float = 80.0,
        # no-ransac options
        grd_feat_width: int = 77,
        sample_mode: str = "topk",   # "topk" | "multinomial"
        irls_iters: int = 0,
    ):
        if yaw_candidates_deg is None:
            if yaw_full360:
                ys = torch.arange(-180.0, 180.0, yaw_deg_step)
            else:
                ys = torch.arange(-10.0, 10.0, yaw_deg_step)
        else:
            ys = torch.tensor(yaw_candidates_deg, dtype=torch.float32)

        self.yaw_candidates = ys * math.pi / 180.0
        self.N_yaw = int(N_yaw)
        self.N = int(num_samples_matches)
        self.th_inlier = float(th_inlier)
        self.th_soft = float(th_soft_inlier)

        self.Hs = int(Hs)
        self.Ws = int(Ws)
        self.sat_bev_res = int(sat_bev_res)
        self.grd_img_res_x = int(grd_img_size_x)
        self.grd_feat_width = int(grd_feat_width)

        self.sample_mode = sample_mode
        self.irls_iters = int(irls_iters)

    @staticmethod
    def _rotate_dirs(u: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        cos = torch.cos(yaw)[..., None]
        sin = torch.sin(yaw)[..., None]
        ux = u[..., 0:1]
        uy = u[..., 1:2]
        ux_r = cos * ux - sin * uy
        uy_r = sin * ux + cos * uy
        return torch.cat([ux_r, uy_r], dim=-1)

    @staticmethod
    def _grid_idx_to_sat_pixels(idx_1d: torch.Tensor, bev_res: int, H: int, W: int):
        r = torch.div(idx_1d, bev_res, rounding_mode="trunc")
        c = idx_1d % bev_res
        y = (r.float() + 0.5) / bev_res * H
        x = (c.float() + 0.5) / bev_res * W
        return y, x

    @staticmethod
    def _pano_dirs_from_indices(
        col_indices: torch.Tensor,
        feat_width: int,
        cam_k: torch.Tensor,
        img_width: int,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        if cam_k.ndim == 2:
            cam_k = cam_k.unsqueeze(0)
        if col_indices.ndim == 1:
            col_indices = col_indices.unsqueeze(0)

        B = col_indices.shape[0]
        if cam_k.shape[0] == 1 and B > 1:
            cam_k = cam_k.expand(B, -1, -1)
        if col_indices.shape[0] == 1 and cam_k.shape[0] > 1:
            col_indices = col_indices.expand(cam_k.shape[0], -1)

        u_pix = (col_indices.float() + 0.5) * (float(img_width) / float(feat_width))
        fx = cam_k[:, 0, 0].unsqueeze(-1).clamp_min(eps)
        cx = cam_k[:, 0, 2].unsqueeze(-1)

        theta = torch.atan((u_pix - cx) / fx)
        ux = torch.sin(theta)  # right
        uy = torch.cos(theta)  # front
        return torch.stack([ux, uy], dim=-1)

    @staticmethod
    def _lines_from_pixels_and_dirs(sat_x: torch.Tensor, sat_y: torch.Tensor, u: torch.Tensor):
        a = u[..., 1]
        b = u[..., 0]
        c = -(a * sat_x + b * sat_y)
        return torch.stack([a, b, c], dim=-1)

    @staticmethod
    def _dist_point_to_lines(p: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
        px = p[..., 0].unsqueeze(-1)
        py = p[..., 1].unsqueeze(-1)
        a = lines[..., 0]
        b = lines[..., 1]
        c = lines[..., 2]
        return (a * px + b * py + c).abs()

    @staticmethod
    def _weighted_ls_point_from_lines(
        lines: torch.Tensor, weights: torch.Tensor, eps: float = 1e-9
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a = lines[..., 0]
        b = lines[..., 1]
        c = lines[..., 2]
        w = weights.clamp_min(eps)

        M11 = (w * a * a).sum(dim=-1)
        M12 = (w * a * b).sum(dim=-1)
        M22 = (w * b * b).sum(dim=-1)
        r1 = -(w * a * c).sum(dim=-1)
        r2 = -(w * b * c).sum(dim=-1)

        M = torch.stack(
            [torch.stack([M11, M12], dim=-1), torch.stack([M12, M22], dim=-1)],
            dim=-2,
        )
        rhs = torch.stack([r1, r2], dim=-1).unsqueeze(-1)

        mean_diag = 0.5 * (M11 + M22)
        lam = (1e-3 * mean_diag).clamp_min(eps)
        I2_shape = [1] * (M.ndim - 2) + [2, 2]
        I2 = torch.eye(2, device=M.device, dtype=M.dtype).view(*I2_shape)
        M_reg = M + lam.unsqueeze(-1).unsqueeze(-1) * I2

        M_pinv = torch.linalg.pinv(M_reg, rcond=1e-6)
        p = (M_pinv @ rhs).squeeze(-1)
        invalid = torch.isnan(p).any(dim=-1) | torch.isinf(p).any(dim=-1)
        return p, invalid

    def _soft_score(
        self, lines: torch.Tensor, p: torch.Tensor, weights: torch.Tensor, th_soft: float
    ) -> torch.Tensor:
        dist = self._dist_point_to_lines(p, lines)
        score = weights / (1.0 + (dist / max(th_soft, 1e-9)) ** 2)
        return score.sum(dim=-1)

    def _build_line_inputs(
        self, matching_score: torch.Tensor, cam_k: torch.Tensor, num_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, Ms, Ng = matching_score.shape
        weights_all = matching_score.reshape(B, Ms * Ng).clamp_min(1e-12)
        N_use = min(int(num_samples), Ms * Ng)

        if self.sample_mode == "topk":
            sampled_idx = torch.topk(weights_all, k=N_use, dim=1).indices
        elif self.sample_mode == "multinomial":
            sampled_idx = torch.multinomial(weights_all, N_use, replacement=False)
        else:
            raise ValueError(f"Unsupported sample_mode: {self.sample_mode}")

        sampled_idx_sat = torch.div(sampled_idx, Ng, rounding_mode="trunc")
        sampled_idx_grd = sampled_idx % Ng
        weights = torch.gather(weights_all, 1, sampled_idx)

        sat_y, sat_x = self._grid_idx_to_sat_pixels(
            sampled_idx_sat, self.sat_bev_res, self.Hs, self.Ws
        )
        grd_x_idx = sampled_idx_grd % self.grd_feat_width
        u_base = self._pano_dirs_from_indices(
            grd_x_idx, self.grd_feat_width, cam_k, self.grd_img_res_x
        )
        return sat_x, sat_y, u_base, weights

    def _solve_translation_given_yaw(
        self,
        sat_x: torch.Tensor,
        sat_y: torch.Tensor,
        u_base: torch.Tensor,
        weights: torch.Tensor,
        yaw_rad: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        u_rot = self._rotate_dirs(u_base, yaw_rad[:, None])
        lines = self._lines_from_pixels_and_dirs(sat_x, sat_y, u_rot)
        p, invalid = self._weighted_ls_point_from_lines(lines, weights)

        for _ in range(self.irls_iters):
            dist = self._dist_point_to_lines(p, lines)
            robust = 1.0 / (1.0 + (dist / max(self.th_inlier, 1e-9)) ** 2)
            p_new, invalid_new = self._weighted_ls_point_from_lines(lines, weights * robust)
            valid_new = ~invalid_new
            p = torch.where(valid_new.unsqueeze(-1), p_new, p)
            invalid = ~valid_new

        score = self._soft_score(lines, p, weights, self.th_soft)
        score = score.masked_fill(invalid, -1e12)
        return p, score

    def estimate_orientation(
        self,
        matching_score: torch.Tensor,
        cam_k: torch.Tensor,
        yaw_chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cam_k is None:
            raise ValueError("cam_k must be provided for KITTI orientation estimation.")

        device = matching_score.device
        dtype = matching_score.dtype
        B = matching_score.shape[0]

        sat_x, sat_y, u_base, weights = self._build_line_inputs(
            matching_score, cam_k, self.N_yaw
        )

        yaw_cands = self.yaw_candidates.to(device=device, dtype=dtype)
        Y = yaw_cands.numel()
        if yaw_chunk is None:
            yaw_chunk = Y

        best_yaw = torch.zeros((B,), device=device, dtype=dtype)
        best_score = torch.full((B,), -1e12, device=device, dtype=dtype)

        for s in range(0, Y, yaw_chunk):
            yaw_blk = yaw_cands[s:s + yaw_chunk]
            Yc = yaw_blk.numel()

            u_rot = self._rotate_dirs(
                u_base.unsqueeze(1), yaw_blk.view(1, Yc, 1)
            )  # (B,Yc,N,2)
            lines = self._lines_from_pixels_and_dirs(
                sat_x.unsqueeze(1), sat_y.unsqueeze(1), u_rot
            )  # (B,Yc,N,3)
            w_expand = weights.unsqueeze(1).expand(B, Yc, weights.shape[-1])  # (B,Yc,N)

            p_blk, invalid_blk = self._weighted_ls_point_from_lines(lines, w_expand)
            score_blk = self._soft_score(lines, p_blk, w_expand, self.th_soft)
            score_blk = score_blk.masked_fill(invalid_blk, -1e12)

            blk_best_score, blk_best_idx = score_blk.max(dim=1)
            blk_best_yaw = yaw_blk[blk_best_idx]

            better = blk_best_score > best_score
            best_score = torch.where(better, blk_best_score, best_score)
            best_yaw = torch.where(better, blk_best_yaw, best_yaw)

        return best_yaw, best_score

    def estimate_pose(
        self,
        matching_score: torch.Tensor,
        matching_score_orig=None,
        cam_k=None,
        return_inliers: bool = False,
    ):
        if cam_k is None:
            raise ValueError("cam_k must be provided for KITTI orientation estimation.")

        best_yaw, _ = self.estimate_orientation(matching_score, cam_k=cam_k, yaw_chunk=None)

        sat_x, sat_y, u_base, weights = self._build_line_inputs(
            matching_score, cam_k, self.N
        )
        p_best, s_best = self._solve_translation_given_yaw(
            sat_x, sat_y, u_base, weights, best_yaw
        )

        return best_yaw, p_best, s_best













def ARC_Solver(
    sampled_sat_idx: torch.Tensor,
    # sat_offset 인자 제거됨
    Hs: int,
    u_ground: torch.Tensor,         # (B,K,2)
    weights: torch.Tensor=None,     # (B,K)
    sat_bev_res: int = 41,
):
    """
    ARC 솔버 (오프셋 없음. pstar와 residual_loss 반환)
    """
    Ws = Hs

    # [수정] 오프셋 없는 원본 grid_idx_to_sat_pixels 함수 호출
    yg, xg = grid_idx_to_sat_pixels(
        idx_1d=sampled_sat_idx,
        bev_res=sat_bev_res,
        H=Hs,
        W=Ws
    )  # (B, N), (B, N) # image coordinates
    
    ug = u_ground  # (B, N, 2) # cartesian coordinates

    # dir vector
    ux = ug[..., 0]
    uy = ug[..., 1] 

    a = uy # img coords : -uy
    b = ux
    c = -(a * xg + b * yg)

    # (B, N, 3)
    lines_t = torch.stack([a, b, c], dim=-1)
    
    # pstar와 residual_loss를 함께 받음
    pstar, residual_loss = least_squares_point_to_lines(lines_t, weights)

    # (B, 2)와 (B,)를 반환
    return pstar, residual_loss



def least_squares_point_to_lines(lines: torch.Tensor, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Least-squares 솔루션과 잔차(residual) Loss를 계산합니다.

    Args:
        lines: (B, N, 3) 또는 (N, 3) 형태의 라인 방정식 [a, b, c] 텐서.
        weights: (B, N) 또는 (N,) 형태의 가중치 텐서 (optional).

    Returns:
        sol: (B, 2) 또는 (2,) 형태의 pstar (최적의 교차점).
        residual_loss: (B,)  
    """
    is_batched = lines.ndim == 3
    if not is_batched:
        lines = lines.unsqueeze(0)  # (1, N, 3)
        if weights is not None and weights.ndim == 1:
            weights = weights.unsqueeze(0)  # (1, N)

    a, b, c = lines[..., 0], lines[..., 1], lines[..., 2] # (B, N)
    
    # 1.0 / (a^2 + b^2)
    d_inv = 1.0 / (a * a + b * b).clamp_min(1e-9) 

    # 기본 노멀라이징 가중치
    w = d_inv # (B, N)

    # 사용자 가중치(weights) 반영
    if weights is not None:
        w = w * weights # (B, N)

    # --- Least-Squares 시스템 구성 ---
    M11 = torch.sum(w * a * a, dim=-1)
    M12 = torch.sum(w * a * b, dim=-1)
    M22 = torch.sum(w * b * b, dim=-1)
    rhs1 = -torch.sum(w * a * c, dim=-1)
    rhs2 = -torch.sum(w * b * c, dim=-1)

    M = torch.stack([
        torch.stack([M11, M12], dim=-1),
        torch.stack([M12, M22], dim=-1)
    ], dim=-2)  # (B, 2, 2)
    rhs = torch.stack([rhs1, rhs2], dim=-1).unsqueeze(-1)  # (B, 2, 1)

    # --- 솔루션 (pstar) 계산 ---
    sol = torch.linalg.solve(M, rhs).squeeze(-1)  # (B, 2)

    # --- 잔차(Residual) Loss 계산 ---
    px = sol[..., 0].unsqueeze(-1)
    py = sol[..., 1].unsqueeze(-1)

    line_errors_sq = (a * px + b * py + c)**2
    dist_sq = line_errors_sq * d_inv # (B, N) 

    if weights is not None:
        rescaled_weights = torch.softmax(weights, dim=-1)
        
        weighted_dist_sq = dist_sq * rescaled_weights
    else:
        weighted_dist_sq = dist_sq
        
    residual_loss = torch.mean(weighted_dist_sq, dim=-1) # (B,) 

    return sol, residual_loss


def grid_idx_to_sat_pixels(idx_1d: torch.Tensor, bev_res: int, H: int, W: int):
    """
    BEV 셀 중심을 위성 이미지 픽셀좌표로 매핑.
    반환: (B,K) y 픽셀, (B,K) x 픽셀 (float)
    """
    r = torch.div(idx_1d, bev_res, rounding_mode='trunc')
    c = idx_1d % bev_res
    y = (r.float() + 0.5) / bev_res * H
    x = (c.float() + 0.5) / bev_res * W
    return y, x # image coordinates



def pano_dirs_from_indices(col_indices: torch.Tensor, pano_width: int) -> torch.Tensor:
    """
    파노라마 이미지의 수평(w) 1D 인덱스로부터 방위 단위벡터 u=(ux,uy)를 계산합니다.
    파노라마의 정중앙 컬럼이 북쪽(North, 0도)을 향합니다.
    (기존 BEV 그리드 로직을 파노라마 로직으로 대체함)

    인자:
    col_indices: (B,K) 또는 (N,) 형태의 픽셀 컬럼(w방향) 인덱스.
                (이전 함수의 'idx_1d'와 동일한 변수명 사용)
    pano_width:  파노라마 이미지의 전체 너비 (w의 길이).
                (이전 함수의 'bev_res'와 동일한 변수명 사용)
    반환:
    (B,K,2) 또는 (N,2) 형태의 (ux, uy) 단위 벡터 (ux=+East, uy=+North)
    """
    
    c = col_indices.float()

    c0 = (pano_width - 1) / 2.0

    displacement = c - c0

    angle = displacement * (2.0 * math.pi / pano_width)

    ux = torch.sin(angle)  # +East component
    uy = torch.cos(angle)  # +North component

    return torch.stack([ux, uy], dim=-1)


def pano_dirs_from_indices_kitti(
    col_indices: torch.Tensor,     # (B,K) or (N,)  "feature-grid x index"
    feat_width: int,               # feature grid width (예: grd_feat_x_size)
    cam_k: torch.Tensor,           # (B,3,3) or (3,3)
    img_width: int,                # resized ground image width (args.ground_image_size[1])
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    KITTI perspective 카메라용 bearing 생성 (K 기반).
    - 이미지 중심(Principal point cx)이 정면(Front, 0 rad)이라고 가정
    - 반환 벡터는 (Right, Front) = (ux, uy) 형태

    Steps:
      1) feature-grid x -> pixel u
      2) theta = atan((u - cx) / fx)
      3) u = [sin(theta), cos(theta)]
    """

    # --- 배치 처리 ---
    if cam_k.ndim == 2:
        cam_k = cam_k.unsqueeze(0)  # (1,3,3)

    # col_indices가 (N,)이면 (1,N)로
    if col_indices.ndim == 1:
        col_indices = col_indices.unsqueeze(0)

    B = cam_k.shape[0]

    # col_indices가 (1,K)인데 B>1이면 broadcast
    if col_indices.shape[0] == 1 and B > 1:
        col_indices = col_indices.expand(B, -1)

    # --- 1) feature x index -> pixel u ---
    # (i + 0.5) * (W_img / W_feat)
    u_pix = (col_indices.float() + 0.5) * (float(img_width) / float(feat_width))  # (B,K)

    # --- 2) intrinsic ---
    fx = cam_k[:, 0, 0].unsqueeze(1).clamp_min(eps)  # (B,1)
    cx = cam_k[:, 0, 2].unsqueeze(1)                 # (B,1)

    # --- 3) horizontal angle in camera frame ---
    theta = torch.atan((u_pix - cx) / fx)            # (B,K)

    # --- 4) unit bearing (Right, Front) ---
    ux = torch.sin(theta)
    uy = torch.cos(theta)

    return torch.stack([ux, uy], dim=-1)             # (B,K,2)

