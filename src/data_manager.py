import os
import glob
import re

class TimeSeriesManager:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.files = []
        self.refresh()

    def refresh(self):
        """
        디렉토리를 스캔하여 LAS 파일을 찾고 시간 순서대로 정렬합니다.
        파일명 형식: MMDDHHMM.las
        """
        all_las = glob.glob(os.path.join(self.dir_path, "*.las"))
        # 파일명을 숫자로 변환하여 정렬 (시간순)
        self.files = sorted(all_las, key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
        print(f":: TimeSeriesManager: Found {len(self.files)} files in {self.dir_path}")
        for i, f in enumerate(self.files):
            print(f"   [{i}] {os.path.basename(f)}")

    def get_baseline(self):
        """가장 첫 번째 파일을 기준(Baseline)으로 반환합니다."""
        return self.files[0] if self.files else None

    def get_comparisons(self):
        """기준 파일을 제외한 나머지 파일들을 반환합니다."""
        return self.files[1:] if len(self.files) > 1 else []

    def get_pairs(self):
        """(이전, 현재) 쌍으로 비교하고 싶을 때 사용합니다."""
        pairs = []
        for i in range(len(self.files) - 1):
            pairs.append((self.files[i], self.files[i+1]))
        return pairs
