from src.loader import load_las_to_o3d
from src.registration import align_point_clouds
from src.processing import separate_background_and_object, smooth_point_cloud_mls
from src.adjustment import RigorousAdjustmentEngine # New Engine
import open3d as o3d
import numpy as np
import os

def main():
    # 1. Load & Initial Align
    source_path = "data/raw/lambo_on_stool.las"
    target_path = "data/raw/stool.las"
    
    print(":: Loading point clouds...")
    target, offset = load_las_to_o3d(target_path)
    if target is None: return
    source, _ = load_las_to_o3d(source_path, global_offset=offset)
    if source is None: return

    # 1-1. Rough Alignment (Open3D RANSAC/ICP)
    # 초기 위치를 잡는 용도입니다.
    print(":: Step 1: Initial Rough Alignment...")
    voxel_size = 0.1
    transformation = align_point_clouds(source, target, voxel_size=voxel_size)
    source.transform(transformation)
    
    # 2. Rigorous Adjustment (LESS Engine)
    # 직접 구현한 디자인 매트릭스 기반 엔진으로 정밀 보정
    print("\n:: Step 2: Rigorous Least Squares Adjustment (Custom Engine)...")
    engine = RigorousAdjustmentEngine(source, target, voxel_size=0.05)
    
    # align() 함수가 내부적으로 source를 변환시킵니다.
    # 반환값: 최종 변환 행렬(Accumulated), 변환된 Source 객체
    final_transform, refined_source = engine.align(max_iterations=10, tolerance=1e-5)
    
    # 3. Separation & Correction
    print("\n:: Step 3: Separation and Correction...")
    pcd_bg, pcd_obj = separate_background_and_object(
        refined_source, target, 
        hue_threshold=0.1, 
        dist_threshold=0.2 
    )
    
    # 4. Smoothing
    print(":: Smoothing Background...")
    pcd_bg_smooth = smooth_point_cloud_mls(pcd_bg, nb_neighbors=20, std_ratio=2.0)
    print(":: Cleaning Object...")
    pcd_obj_clean = smooth_point_cloud_mls(pcd_obj, nb_neighbors=10, std_ratio=1.5)

    # 5. Interactive Visualization
    print("\n" + "="*50)
    print("   INTERACTIVE VIEWER (Rigorous Adjusted)")
    print("   [1] Toggle Background (Stool)")
    print("   [2] Toggle Object (Lamborghini)")
    print("   [Q] Quit")
    print("="*50 + "\n")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Rigorous Adjustment Result", width=1600, height=900)
    
    vis.add_geometry(pcd_bg_smooth)
    vis.add_geometry(pcd_obj_clean)
    
    show_bg = True
    show_obj = True
    
    def toggle_bg(vis):
        nonlocal show_bg
        if show_bg:
            vis.remove_geometry(pcd_bg_smooth, reset_bounding_box=False)
        else:
            vis.add_geometry(pcd_bg_smooth, reset_bounding_box=False)
        show_bg = not show_bg
        return False
        
    def toggle_obj(vis):
        nonlocal show_obj
        if show_obj:
            vis.remove_geometry(pcd_obj_clean, reset_bounding_box=False)
        else:
            vis.add_geometry(pcd_obj_clean, reset_bounding_box=False)
        show_obj = not show_obj
        return False

    vis.register_key_callback(49, toggle_bg)
    vis.register_key_callback(50, toggle_obj)
    
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()