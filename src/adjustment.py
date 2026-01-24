import numpy as np
import open3d as o3d
import copy
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

class RigorousAdjustmentEngine:
    def __init__(self, source, target, voxel_size=0.1):
        self.source = copy.deepcopy(source)
        self.target = target
        self.voxel_size = voxel_size
        
        # Scipy cKDTree for high-speed bulk query
        self.target_points = np.asarray(target.points)
        print(":: Building Scipy cKDTree for Target...")
        self.kdtree = cKDTree(self.target_points)
        
        self.target_normals = np.asarray(target.normals)
        
        # Target Color (HSV 변환) - 가중치 계산용
        if len(target.colors) > 0:
            self.target_hsv = mcolors.rgb_to_hsv(np.asarray(target.colors))
        else:
            self.target_hsv = None

    def find_correspondences(self, current_source, max_dist):
        """
        현재 Source 점들에 대해 Target의 가장 가까운 점(NN)을 대량으로 검색합니다.
        Scipy를 사용하여 매우 빠릅니다.
        """
        src_points = np.asarray(current_source.points)
        
        # Bulk query
        distances, indices = self.kdtree.query(src_points, k=1, workers=-1)
        
        mask = distances < max_dist
        return mask, indices, distances

    def compute_weights(self, src_idx, tgt_idx, current_source, distances):
        """
        가중치 행렬 P의 대각 성분을 계산합니다. (Robust Weighting)
        """
        num_corr = len(src_idx)
        weights = np.ones(num_corr, dtype=np.float64)
        
        curr_src_pts = np.asarray(current_source.points)[src_idx]
        curr_tgt_pts = self.target_points[tgt_idx]
        curr_tgt_norms = self.target_normals[tgt_idx]
        
        # 1. Distance Weight (M-estimator style)
        sigma_dist = self.voxel_size * 0.5 
        dists = distances[src_idx]
        w_dist = np.exp(-(dists**2) / (sigma_dist**2))
        weights *= w_dist
        
        # 2. Normal Similarity Weight
        s_norm = np.asarray(current_source.normals)[src_idx]
        dot = np.sum(s_norm * curr_tgt_norms, axis=1)
        w_normal = np.clip(dot, 0, 1)
        w_normal = w_normal ** 4 
        weights *= w_normal
        
        # 3. Occlusion Check (Directional Check)
        vec = curr_src_pts - curr_tgt_pts
        proj = np.sum(vec * curr_tgt_norms, axis=1)
        
        # Target 표면 아래에 깊게 파묻힌 점 제외
        is_occluded = proj < -(self.voxel_size * 0.2)
        weights[is_occluded] = 0.0
        
        # 4. Color Similarity Weight
        if self.target_hsv is not None:
            s_colors = np.asarray(current_source.colors)[src_idx]
            s_hsv = mcolors.rgb_to_hsv(s_colors)
            t_hsv = self.target_hsv[tgt_idx]
            
            hue_diff = np.abs(s_hsv[:, 0] - t_hsv[:, 0])
            hue_diff = np.minimum(hue_diff, 1.0 - hue_diff)
            
            sigma_color = 0.1
            w_color = np.exp(-(hue_diff**2) / (sigma_color**2))
            weights *= w_color
            
        return weights

    def build_linear_system(self, src_points, tgt_points, tgt_normals, weights):
        cross = np.cross(src_points, tgt_normals)
        A = np.hstack((cross, tgt_normals))
        
        diff = tgt_points - src_points
        L = np.sum(diff * tgt_normals, axis=1)
        
        sqrt_w = np.sqrt(weights)[:, np.newaxis]
        A_weighted = A * sqrt_w
        L_weighted = L * sqrt_w.flatten()
        
        return A_weighted, L_weighted

    def solve_iteration(self, max_dist=0.1):
        mask, tgt_indices, distances = self.find_correspondences(self.source, max_dist)
        
        if np.sum(mask) < 6:
            return None, 0
            
        src_idx = np.where(mask)[0]
        tgt_idx = tgt_indices[mask]
        
        curr_src_pts = np.asarray(self.source.points)[src_idx]
        curr_tgt_pts = self.target_points[tgt_idx]
        curr_tgt_norms = self.target_normals[tgt_idx]
        
        weights = self.compute_weights(src_idx, tgt_idx, self.source, distances)
        
        A, L = self.build_linear_system(curr_src_pts, curr_tgt_pts, curr_tgt_norms, weights)
        x, residuals, rank, s = np.linalg.lstsq(A, L, rcond=None)
        
        return x, np.mean(residuals) if len(residuals) > 0 else 0

    def construct_transform_matrix(self, x):
        alpha, beta, gamma, tx, ty, tz = x
        R = o3d.geometry.get_rotation_matrix_from_xyz((alpha, beta, gamma))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [tx, ty, tz]
        return T

    def align(self, max_iterations=20, tolerance=1e-6):
        current_transform = np.eye(4)
        print(f":: Starting Rigorous Adjustment (Robust Mode, Max Iter: {max_iterations})")
        
        for i in range(max_iterations):
            decay = max(1.0, (10 - i) / 2.0)
            max_dist = self.voxel_size * decay
            
            x, resid = self.solve_iteration(max_dist)
            
            if x is None: break
            
            delta_T = self.construct_transform_matrix(x)
            self.source.transform(delta_T)
            current_transform = delta_T @ current_transform
            
            delta_norm = np.linalg.norm(x)
            print(f"   Iter {i+1}: Delta Norm={delta_norm:.6f}, Mean Residual={resid:.6f}")
            
            if delta_norm < tolerance:
                print(":: Converged.")
                break
                
        return current_transform, self.source
