import open3d as o3d
import numpy as np

def create_mesh_from_pcd(pcd, depth=9):
    """
    Poisson Surface Reconstruction을 사용하여 포인트 클라우드를 메쉬로 변환합니다.
    """
    print(f":: Creating Mesh using Poisson Reconstruction (depth={depth})...")
    
    # 법선 벡터가 필수입니다.
    if not pcd.has_normals():
        print("   - Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Poisson Reconstruction 실행
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    
    # 밀도가 낮은 부분(외곽 노이즈) 제거
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    print(f"   - Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles.")
    return mesh

def save_mesh(mesh, file_path):
    """
    메쉬를 파일로 저장합니다. (.ply, .obj, .stl 등)
    """
    o3d.io.write_triangle_mesh(file_path, mesh)
    print(f":: Mesh saved to {file_path}")
