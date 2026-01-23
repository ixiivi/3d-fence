from src.loader import load_las_to_o3d
from src.registration import align_point_clouds
from src.processing import separate_background_and_object, smooth_point_cloud_mls
import open3d as o3d
import numpy as np
import os

def main():
    # 1. Load & Align
    source_path = "data/raw/lambo_on_stool.las"
    target_path = "data/raw/stool.las"
    
    print(":: Loading point clouds...")
    target, offset = load_las_to_o3d(target_path)
    if target is None: return
    source, _ = load_las_to_o3d(source_path, global_offset=offset)
    if source is None: return

    voxel_size = 0.1
    transformation = align_point_clouds(source, target, voxel_size=voxel_size)
    source.transform(transformation)
    
    # 2. Separation & Correction
    print(":: Separating and Correcting...")
    pcd_bg, pcd_obj = separate_background_and_object(
        source, target, 
        hue_threshold=0.1, 
        dist_threshold=0.2 # 20cm 이상 떨어져 있으면 확실히 다른 물체
    )
    
    # 3. Smoothing (Background Only)
    # 배경(스툴)은 보정 후 스무딩하여 아주 깨끗하게 만듦
    print(":: Smoothing Background...")
    pcd_bg_smooth = smooth_point_cloud_mls(pcd_bg, nb_neighbors=20, std_ratio=2.0)
    
    # 객체(람보르기니)는 노이즈 제거만 살짝 (모양 유지)
    print(":: Cleaning Object...")
    pcd_obj_clean = smooth_point_cloud_mls(pcd_obj, nb_neighbors=10, std_ratio=1.5)

    # 4. Save to PLY
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    bg_path = os.path.join(output_dir, "background_stool.ply")
    obj_path = os.path.join(output_dir, "object_lambo.ply")
    
    print(f":: Saving to {bg_path}...")
    o3d.io.write_point_cloud(bg_path, pcd_bg_smooth)
    print(f":: Saving to {obj_path}...")
    o3d.io.write_point_cloud(obj_path, pcd_obj_clean)
    
    # 5. Interactive Visualization
    print("\n" + "="*50)
    print("   INTERACTIVE VIEWER STARTED")
    print("   [1] Toggle Background (Stool)")
    print("   [2] Toggle Object (Lamborghini)")
    print("   [Q] Quit")
    print("="*50 + "\n")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Change Detection Viewer", width=1600, height=900)
    
    # 회색 배경(Target)도 참조용으로 추가 (선택 사항)
    # target.paint_uniform_color([0.3, 0.3, 0.3])
    # vis.add_geometry(target) 

    # 우리가 만든 두 개의 지오메트리 추가
    vis.add_geometry(pcd_bg_smooth)
    vis.add_geometry(pcd_obj_clean)
    
    # Visibility State
    # Note: Open3D Visualizer doesn't allow simple hide/show. 
    # We must remove_geometry / add_geometry dynamically.
    
    show_bg = True
    show_obj = True
    
    def toggle_bg(vis):
        nonlocal show_bg
        if show_bg:
            vis.remove_geometry(pcd_bg_smooth, reset_bounding_box=False)
            print(">> Background Hidden")
        else:
            vis.add_geometry(pcd_bg_smooth, reset_bounding_box=False)
            print(">> Background Shown")
        show_bg = not show_bg
        return False
        
    def toggle_obj(vis):
        nonlocal show_obj
        if show_obj:
            vis.remove_geometry(pcd_obj_clean, reset_bounding_box=False)
            print(">> Object Hidden")
        else:
            vis.add_geometry(pcd_obj_clean, reset_bounding_box=False)
            print(">> Object Shown")
        show_obj = not show_obj
        return False

    # Key Callbacks (ASCII code: 49='1', 50='2')
    vis.register_key_callback(49, toggle_bg)
    vis.register_key_callback(50, toggle_obj)
    
    # Render Options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.05])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
