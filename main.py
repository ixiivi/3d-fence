from src.visualization import render_high_quality

def main():
    """
    data/raw 폴더에 있는 3D 모델을 렌더링합니다.
    """
    file_path = "data/raw/lambo_on_stool.las"
    render_high_quality(file_path)

if __name__ == "__main__":
    main()
