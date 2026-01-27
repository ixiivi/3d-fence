import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

def preprocess_point_cloud(pcd, voxel_size=0.02):
    print(f":: Voxel downsampling with size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(":: Estimating normals")
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(k=15)
    return pcd_down

def smooth_point_cloud_mls(pcd, nb_neighbors=20, std_ratio=2.0):
    print(f":: Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_cleaned = pcd.select_by_index(ind)
    return pcd_cleaned

def convert_to_hsv(pcd):
    rgb = np.asarray(pcd.colors)
    if len(rgb) == 0: return None
    return mcolors.rgb_to_hsv(rgb)

def separate_background_and_object(source, target, hue_threshold=0.1, dist_threshold=0.1, voxel_size=0.05):
    """
    고급 분류 로직: Geometry + Color(H+S) + Scalable DBSCAN
    """
    print(":: Separating Background and Object (Scalable Mode)...")
    
    # 1. Prepare Data
    tgt_tree = cKDTree(np.asarray(target.points))
    tgt_normals = np.asarray(target.normals)
    if len(tgt_normals) == 0:
        target.estimate_normals()
        tgt_normals = np.asarray(target.normals)
        
    src_points = np.asarray(source.points)
    src_hsv = convert_to_hsv(source)
    tgt_hsv = convert_to_hsv(target)
    
    count = len(src_points)
    is_background = np.zeros(count, dtype=bool)
    corrected_points = src_points.copy()
    
    # 2. 1차 분류 (Distance & Color)
    # Bulk query
    dists, indices = tgt_tree.query(src_points, k=1, workers=-1)
    
    # Vectorized checks would be faster, but let's stick to readable loop or logical indexing for hybrid check
    # Let's try vectorized approach for "potential background"
    
    # Distance check
    mask_dist = dists <= dist_threshold
    
    # Color check (can be complex, iterate only potential candidates)
    # To keep logic simple and robust, we iterate indices where distance is okay
    
    # Pre-calculate candidates
    candidate_indices = np.where(mask_dist)[0]
    
    match_count = 0
    
    # Loop Optimization: Only iterate candidates
    for i in candidate_indices:
        tgt_idx = indices[i]
        
        # Color Check
        if src_hsv is not None and tgt_hsv is not None:
            s_src = src_hsv[i, 1]
            s_tgt = tgt_hsv[tgt_idx, 1]
            if abs(s_src - s_tgt) > 0.3: continue
            
            h_src = src_hsv[i, 0]
            h_tgt = tgt_hsv[tgt_idx, 0]
            hue_diff = abs(h_src - h_tgt)
            if hue_diff > 0.5: hue_diff = 1.0 - hue_diff
            if hue_diff > hue_threshold: continue
            
        # Geometry Correction
        n_tgt = tgt_normals[tgt_idx]
        p_src = src_points[i]
        p_tgt = np.asarray(target.points)[tgt_idx]
        
        vec = p_src - p_tgt
        dist_plane = np.dot(vec, n_tgt)
        
        corrected_points[i] = p_src - (dist_plane * n_tgt)
        is_background[i] = True
        match_count += 1

    # 3. 2차 분류: Scalable DBSCAN Clustering
    print(":: Refining Object Cluster (Scalable DBSCAN)...")
    
    obj_indices = np.where(~is_background)[0]
    num_obj = len(obj_indices)
    
    if num_obj > 0:
        obj_points = src_points[obj_indices]
        
        # A. Downsampling for Speed (Proxy Points)
        # Create a temporary Open3D PCD for voxel downsampling
        tmp_pcd = o3d.geometry.PointCloud()
        tmp_pcd.points = o3d.utility.Vector3dVector(obj_points)
        
        # 다운샘플링 (voxel_size의 2배 정도로 듬성듬성하게)
        # 10만 개 -> 수천 개로 줄임
        proxy_pcd = tmp_pcd.voxel_down_sample(voxel_size=voxel_size * 2)
        proxy_points = np.asarray(proxy_pcd.points)
        
        print(f"   - DBSCAN Input: {num_obj} points -> Downsampled to {len(proxy_points)} proxy points")
        
        if len(proxy_points) > 0:
            # B. Run DBSCAN on Proxy Points
            # eps도 다운샘플링 크기에 맞춰서 키워줌
            db = DBSCAN(eps=voxel_size * 4, min_samples=5).fit(proxy_points)
            proxy_labels = db.labels_
            
            # C. Propagate Labels to Original Points (Nearest Neighbor)
            # KDTree of proxy points
            proxy_tree = cKDTree(proxy_points)
            _, nn_indices = proxy_tree.query(obj_points, k=1, workers=-1)
            
            # 각 원본 점은 가장 가까운 Proxy 점의 라벨을 따라감
            original_labels = proxy_labels[nn_indices]
            
            # D. Filter Noise
            # Label -1 (Noise)인 점들을 구제 시도
            noise_local_mask = (original_labels == -1)
            noise_real_indices = obj_indices[noise_local_mask]
            
            reclaimed_count = 0
            for idx in noise_real_indices:
                tgt_idx = indices[idx] # 이미 구해둔 NN
                dist = dists[idx]
                
                # 거리가 아주 멀지 않다면 Background로 편입 (구제)
                # (원래 Background 기준보다는 조금 관대하게)
                if dist < dist_threshold * 1.5:
                    n_tgt = tgt_normals[tgt_idx]
                    p_src = src_points[idx]
                    p_tgt = np.asarray(target.points)[tgt_idx]
                    
                    vec = p_src - p_tgt
                    dist_plane = np.dot(vec, n_tgt)
                    
                    corrected_points[idx] = p_src - (dist_plane * n_tgt)
                    is_background[idx] = True
                    reclaimed_count += 1
            
            print(f"   - Reclaimed {reclaimed_count} noise points to background.")
    
    print(f":: Final Classification - Background: {np.sum(is_background)}, Object: {count - np.sum(is_background)}")

    # 4. Construct Result PCDs
    pcd_bg = o3d.geometry.PointCloud()
    pcd_bg.points = o3d.utility.Vector3dVector(corrected_points[is_background])
    src_colors = np.asarray(source.colors)
    if len(src_colors) > 0:
        pcd_bg.colors = o3d.utility.Vector3dVector(src_colors[is_background])
    src_normals = np.asarray(source.normals)
    if len(src_normals) > 0:
        pcd_bg.normals = o3d.utility.Vector3dVector(src_normals[is_background])
        
    pcd_obj = o3d.geometry.PointCloud()
    pcd_obj.points = o3d.utility.Vector3dVector(src_points[~is_background])
    if len(src_colors) > 0:
        pcd_obj.colors = o3d.utility.Vector3dVector(src_colors[~is_background])
    if len(src_normals) > 0:
        pcd_obj.normals = o3d.utility.Vector3dVector(src_normals[~is_background])
        
    return pcd_bg, pcd_obj
