import open3d as o3d
import numpy as np
import copy
from src.adjustment import RigorousAdjustmentEngine
from src.meshing import create_mesh_from_pcd
from src.analysis import compute_point_to_mesh_distance
from src.registration import align_point_clouds

class GlobalDynamicMap:
    def __init__(self, voxel_size=0.02):
        self.voxel_size = voxel_size
        self.global_pcd = o3d.geometry.PointCloud()
        self.global_mesh = None
        self.accumulated_count = 0
        
        # 순차 정합을 위한 변수들
        self.prev_pcd = None
        self.accumulated_transform = np.identity(4)

    def initialize(self, pcd):
        print(":: Initializing Global Map...")
        self.global_pcd = copy.deepcopy(pcd)
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
        self.update_mesh()
        
        self.prev_pcd = copy.deepcopy(pcd) # 법선 포함
        # 만약 법선이 없다면 계산해둬야 다음 정합에 유리함
        if not self.prev_pcd.has_normals():
            self.prev_pcd.estimate_normals()
            
        self.accumulated_count = 1

    def update_mesh(self):
        if len(self.global_pcd.points) < 100: return
        try:
            self.global_mesh = create_mesh_from_pcd(self.global_pcd, depth=8)
        except Exception as e:
            print(f"   [Map Warning] Mesh update failed: {e}")

    def process_new_scan(self, new_pcd):
        """
        Sequential Alignment -> Change Detection -> Map Update
        """
        # 법선 계산 (필수)
        if not new_pcd.has_normals():
            print("   - Estimating normals for new scan...")
            new_pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            new_pcd.orient_normals_consistent_tangent_plane(k=15)

        if self.accumulated_count == 0:
            self.initialize(new_pcd)
            return o3d.geometry.PointCloud(), new_pcd, None

        print(f":: Processing Scan #{self.accumulated_count + 1}...")

        # 1. Sequential Alignment (t -> t-1)
        # 이전 프레임과 먼저 맞춥니다. 중첩 확률이 가장 높기 때문입니다.
        print("   - Step 1: Aligning to Previous Scan...")
        
        # RANSAC (Global) -> ICP (Fine)
        # voxel_size를 좀 키워서 큰 특징 위주로 잡습니다.
        rel_transform = align_point_clouds(new_pcd, self.prev_pcd, voxel_size=0.1)
        
        # 2. Update Global Pose
        # Global_T_new = Global_T_prev * Prev_T_new
        # 여기서 rel_transform은 new를 prev로 보내는 행렬이므로
        # new_in_global = accumulated_transform * (rel_transform * new)
        # 순서 주의: Open3D transform은 pcd.transform(T) -> T @ p
        
        # 현재 new_pcd를 prev_pcd 좌표계로 변환
        aligned_to_prev = copy.deepcopy(new_pcd)
        aligned_to_prev.transform(rel_transform)
        
        # 이제 prev_pcd는 이미 Global 좌표계에 있다고 가정하지 않고, 
        # accumulated_transform을 통해 Global로 보냄.
        # 하지만 prev_pcd 변수 자체는 이전 루프에서 이미 Global 좌표계로 변환된 상태로 저장하는 것이 편함.
        
        # 전략 수정:
        # self.prev_pcd는 항상 'Global 좌표계로 변환된 직전 프레임'을 저장한다.
        # 그러면 rel_transform만 잘 구하면 바로 Global로 갈 수 있다.
        
        # rel_transform: New(Local) -> Prev(Global)
        # align_point_clouds(source=new, target=prev)
        
        # 변환 적용 (Global 좌표계로 진입)
        new_pcd.transform(rel_transform) 
        aligned_scan = new_pcd
        
        # 3. Fine-tuning with Global Map (Optional but recommended)
        # 누적 오차를 줄이기 위해 Global Map 전체와 살짝 더 맞춤
        # 단, Global Map이 너무 커지면 느려지므로 Voxel Downsample 된 것과 비교
        print("   - Step 2: Fine-tuning with Global Map...")
        engine = RigorousAdjustmentEngine(aligned_scan, self.global_pcd, voxel_size=0.05)
        # 아주 조금만 움직이도록 제한 (max_iter 줄임)
        delta_T, aligned_scan = engine.align(max_iterations=5, tolerance=1e-4)
        
        # 4. Change Detection
        if self.global_mesh is None: self.update_mesh()
        distances = compute_point_to_mesh_distance(aligned_scan, self.global_mesh)
        
        threshold = 0.02
        change_mask = distances > threshold
        
        change_pcd = aligned_scan.select_by_index(np.where(change_mask)[0])
        background_part = aligned_scan.select_by_index(np.where(~change_mask)[0])
        
        print(f"   - Detected {len(change_pcd.points)} change points.")
        
        # 5. Map Update & Save State
        self.global_pcd += background_part
        self.global_pcd = self.global_pcd.voxel_down_sample(self.voxel_size)
        self.update_mesh()
        
        # 다음 턴을 위해 현재(정합된) 스캔을 prev_pcd로 저장
        self.prev_pcd = copy.deepcopy(aligned_scan)
        self.accumulated_count += 1
        
        return change_pcd, aligned_scan, distances