import open3d as o3d
import os
import glob
import re
import numpy as np

def main():
    result_dir = "data/results_dune_v3"
    
    # 결과 파일 찾기 (timestamp_Detected_Change.ply 형식)
    all_ply = glob.glob(os.path.join(result_dir, "*.ply"))
    # final_global_map.ply와 final_global_mesh.ply는 제외
    heatmap_files = sorted([f for f in all_ply if "final_global_" not in f])
    # Global Map 찾기
    map_file = os.path.join(result_dir, "final_global_map.ply")
    
    if not heatmap_files:
        print(f"No result files found in {result_dir}")
        return
        
    print(f":: Found {len(heatmap_files)} time-series results.")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Time-series Viewer", width=1280, height=720)
    
    current_idx = 0
    show_map = False
    
    # Load Global Map (Optional)
    global_map_pcd = None
    if os.path.exists(map_file):
        global_map_pcd = o3d.io.read_point_cloud(map_file)
        global_map_pcd.paint_uniform_color([0.8, 0.8, 0.8]) # Light Gray
    
    def update_view(vis):
        nonlocal current_idx, show_map
        
        vis.clear_geometries()
        
        # Load Current Heatmap
        filename = heatmap_files[current_idx]
        pcd = o3d.io.read_point_cloud(filename)
        vis.add_geometry(pcd, reset_bounding_box=False)
        
        # Add Global Map overlay
        if show_map and global_map_pcd:
            vis.add_geometry(global_map_pcd, reset_bounding_box=False)
            
        print(f">> Showing [{current_idx+1}/{len(heatmap_files)}] {os.path.basename(filename)}")
        return False

    def next_step(vis):
        nonlocal current_idx
        current_idx = (current_idx + 1) % len(heatmap_files)
        update_view(vis)
        return False
        
    def prev_step(vis):
        nonlocal current_idx
        current_idx = (current_idx - 1) % len(heatmap_files)
        update_view(vis)
        return False
        
    def toggle_map(vis):
        nonlocal show_map
        show_map = not show_map
        print(f">> Global Map Overlay: {'ON' if show_map else 'OFF'}")
        update_view(vis)
        return False

    # Key Callbacks
    vis.register_key_callback(262, next_step) # Right
    vis.register_key_callback(263, prev_step) # Left
    vis.register_key_callback(32, toggle_map) # Space
    
    # Initialize
    first_pcd = o3d.io.read_point_cloud(heatmap_files[0])
    vis.add_geometry(first_pcd)
    
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.point_size = 3.0
    
    print("\n[Controls] Right: Next, Left: Prev, Space: Toggle Map Overlay")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
