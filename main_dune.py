from src.loader import load_las_to_o3d
from src.analysis import colorize_by_distance
from src.ai_labeling import GeminiLabeler
from src.data_manager import TimeSeriesManager
from src.mapping import GlobalDynamicMap
import open3d as o3d
import numpy as np
import os
import time

# 사용자 설정
USE_AI = False

def capture_and_analyze(change_pcd, full_pcd, output_dir, file_name, labeler):
    """AI 분석 및 결과 저장"""
    if len(change_pcd.points) < 500:
        return "No Change"

    capture_path = os.path.join(output_dir, f"capture_{file_name}.png")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Analysing {file_name}", width=800, height=800, visible=True)
    vis.add_geometry(full_pcd)
    
    view_ctl = vis.get_view_control()
    view_ctl.set_lookat(change_pcd.get_center())
    view_ctl.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(capture_path, do_render=True)
    vis.destroy_window()
    
    label = "Detected Change"
    if USE_AI and labeler:
        label = labeler.identify_object([capture_path])
        
    return label

def main():
    manager = TimeSeriesManager("data/raw_dune")
    output_dir = "data/results_dune_v3" # 버전 업
    os.makedirs(output_dir, exist_ok=True)
    
    labeler = GeminiLabeler() if USE_AI else None
    
    # Global Map 초기화
    global_map = GlobalDynamicMap(voxel_size=0.02)
    
    history = []
    
    # 모든 파일 순회
    for i, file_path in enumerate(manager.files):
        file_name = os.path.basename(file_path).split('.')[0]
        print(f"\n>>> Processing [{i+1}/{len(manager.files)}]: {file_name} <<<")
        
        # Load (Offset은 첫 파일 기준으로 통일하거나, 
        # Global Map이 중심을 잡고 있으니 개별 로드 시 자체 mean을 빼도 무방하나,
        # 안전하게 첫 파일 Offset 사용)
        if i == 0:
            pcd, offset = load_las_to_o3d(file_path)
            global_offset = offset
        else:
            pcd, _ = load_las_to_o3d(file_path, global_offset=global_offset)
            
        if pcd is None: continue
        
        # Core Process
        if i == 0:
            # 첫 파일은 무조건 지도로 등록
            global_map.initialize(pcd)
            label = "Baseline (Initialized)"
        else:
            # 이후 파일은 지도와 비교 & 업데이트
            change_pcd, aligned_scan, distances = global_map.process_new_scan(pcd)
            
            # 분석
            label = capture_and_analyze(change_pcd, aligned_scan, output_dir, file_name, labeler)
            
            # 결과 저장 (히트맵)
            heatmap_pcd = colorize_by_distance(aligned_scan, distances, max_dist=0.02)
            safe_label = label.replace(' ', '_')
            o3d.io.write_point_cloud(os.path.join(output_dir, f"{file_name}_{safe_label}.ply"), heatmap_pcd)

        history.append({"file": file_name, "label": label})
        
    # Final Map 저장
    print("\n:: Saving Final Global Map...")
    o3d.io.write_point_cloud(os.path.join(output_dir, "final_global_map.ply"), global_map.global_pcd)
    if global_map.global_mesh:
        o3d.io.write_triangle_mesh(os.path.join(output_dir, "final_global_mesh.ply"), global_map.global_mesh)

    # Summary
    print("\n" + "="*50)
    print("   DYNAMIC MAP ANALYSIS SUMMARY")
    for entry in history:
        print(f"   [{entry['file']}] -> {entry['label']}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
