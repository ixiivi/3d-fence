import open3d as o3d
import numpy as np
import laspy
import os

def render_high_quality(file_path):
    """
    .las 파일을 고품질로 렌더링하고 시각화합니다.
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다 -> {file_path}")
        return

    print(f"Reading {file_path} for High-Quality rendering...")
    las = laspy.read(file_path)

    # 1. 좌표 데이터 및 중심점 이동 (Precision Issue 해결)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    center = np.mean(points, axis=0)
    points -= center
    print(f"Offset applied (Centering): {center}")

    # 2. 색상 데이터 정규화 (16-bit vs 8-bit)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        max_val = np.max(colors)
        if max_val > 255: # Assuming 16-bit color if max value is greater than 255
            colors = colors / 65535.0
            print("Detected 16-bit color and normalized.")
        else: # Assuming 8-bit color
            colors = colors / 255.0
            print("Detected 8-bit color and normalized.")
    else:
        colors = None # No color data, Open3D will use a default color
        print("No color information found. Using default color.")


    # 3. Open3D 객체 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        pcd.paint_uniform_color([0.6, 0.6, 0.6])


    # 4. 법선 데이터(Normals) 계산 -> 입체감 부여
    print("Estimating normals for lighting effects...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_to_align_with_direction()

    # 5. 시각화
    print("Launching Stable High-Quality Viewer...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D-Fence HQ (2023144077)", width=1600, height=900)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.light_on = True
    vis.run()
    vis.destroy_window()
