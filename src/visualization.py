import open3d as o3d
import numpy as np
import copy

def visualize_pcd(pcd, window_name="Point Cloud"):
    """
    이미 메모리에 로드된 Open3D PointCloud 객체를 시각화합니다.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1600, height=900)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    opt.light_on = True 
    
    vis.run()
    vis.destroy_window()

def draw_registration_result(source, target, transformation):
    """
    정합 결과를 시각화합니다.
    Source는 붉은색, Target은 청록색(Cyan)으로 표시하여 겹침 상태를 확인합니다.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    
    source_temp.paint_uniform_color([1, 0.706, 0])  # Yellow/Orange
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Cyan
    
    source_temp.transform(transformation)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Registration Result (Yellow: Source, Cyan: Target)", width=1600, height=900)
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    vis.run()
    vis.destroy_window()