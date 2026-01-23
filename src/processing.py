import open3d as o3d
import numpy as np
import matplotlib.colors as mcolors

def preprocess_point_cloud(pcd, voxel_size=0.02):
    """
    포인트 클라우드를 다운샘플링하고 법선(Normal)을 추정합니다.
    """
    print(f":: Voxel downsampling with size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)

    print(":: Estimating normals (Hyrid Search)")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    pcd_down.orient_normals_consistent_tangent_plane(k=15)
    return pcd_down

def smooth_point_cloud_mls(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    Statistical Outlier Removal을 사용하여 노이즈를 제거합니다.
    """
    print(f":: Removing outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    pcd_cleaned = pcd.select_by_index(ind)
    return pcd_cleaned

def paint_by_normals(pcd):
    print(":: Painting points based on Normals")
    normals = np.asarray(pcd.normals)
    colors = (normals + 1) / 2
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def convert_to_hsv(pcd):
    rgb = np.asarray(pcd.colors)
    if len(rgb) == 0:
        return None
    hsv = mcolors.rgb_to_hsv(rgb)
    return hsv

def separate_background_and_object(source, target, hue_threshold=0.1, dist_threshold=0.1):
    """
    Source 데이터를 Target(배경)과 비교하여 두 그룹으로 분리합니다.
    
    Returns:
        pcd_background (Open3D PointCloud): Target과 일치하여 보정된 점들 (스툴)
        pcd_object (Open3D PointCloud): 일치하지 않아 원본 유지된 점들 (람보르기니)
    """
    print(":: Separating Background (Corrected) and Object (Original)...")
    
    # 1. KDTree Build
    pcd_tree = o3d.geometry.KDTreeFlann(target)
    
    src_points = np.asarray(source.points)
    tgt_points = np.asarray(target.points)
    tgt_normals = np.asarray(target.normals)
    
    src_hsv = convert_to_hsv(source)
    tgt_hsv = convert_to_hsv(target)
    
    if len(tgt_normals) == 0:
        target.estimate_normals()
        tgt_normals = np.asarray(target.normals)
    
    count = len(src_points)
    
    # 결과를 저장할 리스트 (또는 불리언 마스크)
    is_background = np.zeros(count, dtype=bool) # False로 초기화 (기본값: Object)
    corrected_points = src_points.copy()
    
    # 2. Iterate and Classify
    # (추후 벡터화 가능)
    match_count = 0
    
    for i in range(count):
        # 1. Search NN
        [k, idx, _] = pcd_tree.search_knn_vector_3d(src_points[i], 1)
        tgt_idx = idx[0]
        
        # 2. Check Conditions
        # A. 거리 체크 (너무 멀면 다른 물체)
        dist = np.linalg.norm(src_points[i] - tgt_points[tgt_idx])
        if dist > dist_threshold:
            continue # Background 아님 -> Object로 분류
            
        # B. 색상 체크
        h_src = src_hsv[i, 0]
        h_tgt = tgt_hsv[tgt_idx, 0]
        hue_diff = abs(h_src - h_tgt)
        if hue_diff > 0.5: hue_diff = 1.0 - hue_diff
        
        if hue_diff > hue_threshold:
            continue # 색상 다름 -> Object로 분류
            
        # 3. Correction (Point-to-Plane) -> Background로 분류된 점만 보정
        n_tgt = tgt_normals[tgt_idx]
        p_src = src_points[i]
        p_tgt = tgt_points[tgt_idx]
        
        vec = p_src - p_tgt
        dist_plane = np.dot(vec, n_tgt)
        
        # 투영 (Correction)
        corrected_points[i] = p_src - (dist_plane * n_tgt)
        is_background[i] = True
        match_count += 1

    print(f":: Classification Result - Background: {match_count}, Object: {count - match_count}")
    
    # 3. Create Point Clouds
    # Background (Corrected points)
    pcd_background = o3d.geometry.PointCloud()
    pcd_background.points = o3d.utility.Vector3dVector(corrected_points[is_background])
    # 색상과 법선도 해당 인덱스만 가져옴
    src_colors = np.asarray(source.colors)
    src_normals = np.asarray(source.normals)
    if len(src_colors) > 0:
        pcd_background.colors = o3d.utility.Vector3dVector(src_colors[is_background])
    if len(src_normals) > 0:
        pcd_background.normals = o3d.utility.Vector3dVector(src_normals[is_background])
        
    # Object (Original points)
    # 중요: Object는 corrected_points가 아니라 원본 src_points를 써야 함 (보정 안 된 원형 유지)
    # 하지만 위 로직에서 is_background=False인 점들은 corrected_points[i]를 건드리지 않았으므로 그대로임.
    # 명확성을 위해 src_points 사용
    pcd_object = o3d.geometry.PointCloud()
    pcd_object.points = o3d.utility.Vector3dVector(src_points[~is_background])
    if len(src_colors) > 0:
        pcd_object.colors = o3d.utility.Vector3dVector(src_colors[~is_background])
    if len(src_normals) > 0:
        pcd_object.normals = o3d.utility.Vector3dVector(src_normals[~is_background])
        
    return pcd_background, pcd_object
