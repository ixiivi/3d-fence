from src.loader import load_las_to_o3d
from src.registration import align_point_clouds
from src.adjustment import RigorousAdjustmentEngine
from src.meshing import create_mesh_from_pcd
from src.analysis import compute_point_to_mesh_distance, colorize_by_distance
from src.ai_labeling import GeminiLabeler
import open3d as o3d
import numpy as np
import os
import time
import copy

def capture_multiview(pcd, output_dir, prefix="capture"):
    """
    포인트 클라우드를 Front, Side, Top 3방향에서 캡처합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Capturing for AI...", width=1024, height=1024, visible=True)
    vis.add_geometry(pcd)
    
    opt = vis.get_render_option()
    opt.point_size = 1.0 # 점 크기 축소 (고해상도 느낌)
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    ctr = pcd.get_center()
    view_ctl = vis.get_view_control()
    
    # Views: [Eye, Up, LookAt] -> 하지만 Open3D ViewControl은 rotate/zoom 방식임.
    # set_front, set_up, set_zoom 등을 활용하거나 set_lookat을 씀.
    
    # 1. Front View (기본)
    view_ctl.set_lookat(ctr)
    view_ctl.set_zoom(0.6)
    view_ctl.set_front([0, -1, 0.5]) # 약간 위에서 정면
    view_ctl.set_up([0, 0, 1])
    
    for _ in range(10): # 렌더링 안정화 대기
        vis.poll_events()
        vis.update_renderer()
        
    path1 = os.path.join(output_dir, f"{prefix}_front.png")
    vis.capture_screen_image(path1, do_render=True)
    paths.append(path1)
    
    # 2. Top View
    view_ctl.set_front([0, 0, 1]) # 위에서 아래로
    view_ctl.set_up([0, 1, 0])
    
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
        
    path2 = os.path.join(output_dir, f"{prefix}_top.png")
    vis.capture_screen_image(path2, do_render=True)
    paths.append(path2)
    
    # 3. Side View
    view_ctl.set_front([1, 0, 0.2]) # 측면
    view_ctl.set_up([0, 0, 1])
    
    for _ in range(10):
        vis.poll_events()
        vis.update_renderer()
        
    path3 = os.path.join(output_dir, f"{prefix}_side.png")
    vis.capture_screen_image(path3, do_render=True)
    paths.append(path3)
    
    vis.destroy_window()
    return paths

def main():
    # 1. Load Files
    source_path = "data/raw/lambo_on_stool.las"
    target_path = "data/raw/stool.las"
    
    print(":: Loading point clouds...")
    target_pcd, offset = load_las_to_o3d(target_path)
    source_pcd, _ = load_las_to_o3d(source_path, global_offset=offset)
    
    if target_pcd is None or source_pcd is None: return

    # 2. Registration
    print(":: Step 1: Alignment...")
    voxel_size = 0.1
    trans_rough = align_point_clouds(source_pcd, target_pcd, voxel_size=voxel_size)
    source_pcd.transform(trans_rough)
    
    engine = RigorousAdjustmentEngine(source_pcd, target_pcd, voxel_size=0.05)
    final_transform, refined_source = engine.align(max_iterations=5)
    
    # 3. Reference Mesh
    print("\n:: Step 2: Creating Reference Mesh (Target)...")
    target_pcd_clean = target_pcd.voxel_down_sample(0.02)
    try:
        ref_mesh = create_mesh_from_pcd(target_pcd_clean, depth=8)
    except:
        return
    
    # 4. Change Detection (High Precision: 2cm)
    print("\n:: Step 3: Mesh-based Change Detection (Threshold: 2cm)...")
    distances = compute_point_to_mesh_distance(refined_source, ref_mesh)
    change_mask = distances > 0.02
    
    # 변화된 부분만 추출 (for AI capture)
    # 원본 색상 유지를 위해 refined_source에서 추출
    change_pcd = refined_source.select_by_index(np.where(change_mask)[0])
    
    if len(change_pcd.points) < 100:
        print(":: No significant change detected.")
        return

    # 5. AI Labeling Pipeline (Capture Original RGB)
    print("\n:: Step 4: Multi-view Capture & AI Labeling...")
    
    # 캡처용 객체: 변화된 부분만 + 원본 색상
    # 배경 노이즈 제거 (Optional: DBSCAN could be used here too)
    
    image_paths = capture_multiview(change_pcd, "data/output", prefix="lambo")
    print(f":: Captured {len(image_paths)} views.")
    
    labeler = GeminiLabeler()
    label_text = labeler.identify_object(image_paths)
    
    print(f"\n" + "="*50)
    print(f"   AI ANALYSIS RESULT: [{label_text}]")
    print(f"   Change Detected: {len(change_pcd.points)} points")
    print("="*50 + "\n")
    
    # 6. Final Visualization (with Heatmap)
    heatmap_pcd = colorize_by_distance(refined_source, distances, max_dist=0.02)
    
    vis_final = o3d.visualization.Visualizer()
    vis_final.create_window(window_name=f"Result: {label_text} (Red Area)", width=1280, height=720)
    vis_final.add_geometry(heatmap_pcd)
    
    opt = vis_final.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 2.0
    
    vis_final.run()
    vis_final.destroy_window()

if __name__ == "__main__":
    main()
