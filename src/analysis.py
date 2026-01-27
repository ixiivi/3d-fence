import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def compute_point_to_mesh_distance(pcd, mesh):
    """
    포인트 클라우드의 각 점이 메쉬 표면으로부터 떨어진 거리를 계산합니다.
    """
    print(":: Computing Point-to-Mesh distances...")
    
    # Open3D의 RaycastingScene을 사용하여 거리 계산
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    _ = scene.add_triangles(mesh_t)
    
    # 포인트 데이터를 텐서로 변환
    points_t = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.Dtype.Float32)
    
    # 최단 거리 계산 (unsigned distance)
    # signed_distance를 원한다면 compute_distance를 쓰면 됨
    distances = scene.compute_distance(points_t).numpy()
    
    return distances

def colorize_by_distance(pcd, distances, max_dist=0.1):
    """
    거리에 따라 포인트 클라우드에 히트맵 색상을 입힙니다.
    거리가 가까우면(0) 초록색, 멀면(max_dist 이상) 빨간색.
    """
    print(f":: Colorizing by distance (max_threshold={max_dist})...")
    
    # 거리를 0~1 사이로 정규화
    normalized_dist = np.clip(distances / max_dist, 0, 1)
    
    # Matplotlib의 colormap 적용 (jet: blue-green-red)
    cmap = plt.get_cmap('jet')
    colors = cmap(normalized_dist)[:, :3] # RGBA -> RGB
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
