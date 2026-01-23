import open3d as o3d
import numpy as np
import copy

def preprocess_for_registration(pcd, voxel_size):
    """
    정합을 위해 다운샘플링, 법선 추정, FPFH 특징점 추출을 수행합니다.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(k=15)

    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    RANSAC 기반의 Global Registration을 수행하여 대략적인 위치를 맞춥니다.
    """
    distance_threshold = voxel_size * 1.5
    print(f":: RANSAC alignment (Distance Threshold: {distance_threshold:.3f})")
    
    # RANSAC 반복 횟수와 검증 기준을 강화하여 안정성 확보
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 1000))
    return result

def refine_registration_icp(source, target, voxel_size, init_transform=np.identity(4)):
    """
    ICP (Iterative Closest Point) 알고리즘으로 정밀 정합을 수행합니다.
    """
    distance_threshold = voxel_size * 0.4
    print(f":: ICP Refinement (Distance Threshold: {distance_threshold:.3f})")
    
    if not source.has_normals():
        source.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    if not target.has_normals():
        target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    
    return result

def align_point_clouds(source, target, voxel_size=0.1): # 기본값을 0.1로 증가
    """
    전체 정합 프로세스 실행 (Preprocess -> Global -> ICP)
    """
    print(f":: Preprocessing with voxel size {voxel_size}")
    source_down, source_fpfh = preprocess_for_registration(source, voxel_size)
    target_down, target_fpfh = preprocess_for_registration(target, voxel_size)
    
    # 1. Global Registration (Rough Alignment)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(f"Global Registration Fitness: {result_ransac.fitness:.4f}")
    
    # 2. ICP Refinement (Fine Alignment)
    # RANSAC 결과를 초기값으로 사용
    result_icp = refine_registration_icp(source, target, voxel_size, result_ransac.transformation)
    print(f"ICP Refinement Fitness: {result_icp.fitness:.4f}")
    print(f"Transformation Matrix:\n{result_icp.transformation}")
    
    return result_icp.transformation
