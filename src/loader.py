import laspy
import numpy as np
import open3d as o3d
import os

def load_las_to_o3d(file_path, global_offset=None):
    """
    LAS 파일을 읽어서 Open3D PointCloud 객체로 반환합니다.
    
    :param file_path: 파일 경로
    :param global_offset: 좌표 이동을 위한 오프셋 (numpy array [x, y, z]). 
                          None이면 현재 파일의 중심을 기준으로 이동하고 그 값을 반환합니다.
    :return: (pcd, offset) 튜플을 반환합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return None, None
        
    print(f"Loading {file_path}...")
    las = laspy.read(file_path)
    
    # 좌표 추출
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    # 중심점 계산 및 이동
    if global_offset is None:
        global_offset = np.mean(points, axis=0)
        print(f":: Calculated new centering offset: {global_offset}")
    else:
        print(f":: Applying provided offset: {global_offset}")
        
    points -= global_offset
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 기존 색상 정보 처리
    if hasattr(las, 'red'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose().astype(np.float64)
        max_val = np.max(colors)
        if max_val > 255:
            colors /= 65535.0
        else:
            colors /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
    return pcd, global_offset