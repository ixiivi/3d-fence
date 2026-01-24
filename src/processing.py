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
    고급 분류 로직: Geometry + Color(H+S) + Clustering(DBSCAN)
    """
    print(":: Separating Background and Object (Advanced Mode)...")
    
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
    # Bulk query for speed
    dists, indices = tgt_tree.query(src_points, k=1, workers=-1)
    
    for i in range(count):
        tgt_idx = indices[i]
        dist = dists[i]
        
        # Condition A: Distance
        if dist > dist_threshold:
            continue # Object
            
        # Condition B: Color (Hue & Saturation)
        # 스툴은 회색(Low Saturation), 람보는 유채색(High Saturation)일 가능성 높음
        if src_hsv is not None and tgt_hsv is not None:
            s_src = src_hsv[i, 1] # Saturation
            s_tgt = tgt_hsv[tgt_idx, 1]
            
            # Saturation 차이가 크면 다른 물체 (예: 회색 vs 노랑)
            sat_diff = abs(s_src - s_tgt)
            if sat_diff > 0.3: # 채도 차이가 30% 이상이면
                 continue # Object
            
            h_src = src_hsv[i, 0]
            h_tgt = tgt_hsv[tgt_idx, 0]
            hue_diff = abs(h_src - h_tgt)
            if hue_diff > 0.5: hue_diff = 1.0 - hue_diff
            
            if hue_diff > hue_threshold:
                continue # Object
        
        # Condition C: Normal (Geometry)
        n_tgt = tgt_normals[tgt_idx]
        p_src = src_points[i]
        p_tgt = np.asarray(target.points)[tgt_idx]
        
        # Projection Correction
        vec = p_src - p_tgt
        dist_plane = np.dot(vec, n_tgt)
        
        corrected_points[i] = p_src - (dist_plane * n_tgt)
        is_background[i] = True

    # 3. 2차 분류: DBSCAN Clustering (Refinement)
    # Object로 분류된 점들 중에서 "노이즈(작은 점들)"를 걸러내고, "진짜 덩어리"만 남김
    print(":: Refining Object Cluster (DBSCAN)...")
    
    obj_indices = np.where(~is_background)[0]
    if len(obj_indices) > 0:
        obj_points = src_points[obj_indices]
        
        # DBSCAN: eps=거리, min_samples=최소 점 개수
        # 람보르기니 같은 덩어리는 밀도가 높아야 함
        db = DBSCAN(eps=voxel_size * 2, min_samples=20).fit(obj_points)
        labels = db.labels_
        
        # Label -1은 노이즈
        noise_mask = (labels == -1)
        
        # 노이즈로 판명된 점들은 다시 Background로 보낼지, 아니면 아예 버릴지 결정
        # 여기서는 "Background 후보"로 격상시키거나, 그냥 삭제.
        # 안전하게: Background로 편입 시도 (보정 적용)
        
        noise_real_indices = obj_indices[noise_mask]
        
        # 노이즈 점들에 대해 다시 가까운 Target 평면으로 붙이기 시도
        for idx in noise_real_indices:
            tgt_idx = indices[idx]
            n_tgt = tgt_normals[tgt_idx]
            p_src = src_points[idx]
            p_tgt = np.asarray(target.points)[tgt_idx]
            
            vec = p_src - p_tgt
            dist_plane = np.dot(vec, n_tgt)
            
            # 거리가 아주 멀지 않다면 Background로 구제
            if abs(dist_plane) < dist_threshold:
                corrected_points[idx] = p_src - (dist_plane * n_tgt)
                is_background[idx] = True # 구제 성공
    
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