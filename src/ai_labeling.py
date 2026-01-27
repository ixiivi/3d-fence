import google.generativeai as genai
import os
from PIL import Image

class GeminiLabeler:
    def __init__(self):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("[Warning] GOOGLE_API_KEY not found.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            # 마지막 시도: 가장 기본 모델명 사용
            try:
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print(":: Gemini 1.5 Flash Model Loaded.")
            except:
                self.model = None

    def identify_object(self, image_paths):
        if self.model is None:
            return "Yellow Toy Car (Simulated - No API)"
            
        if isinstance(image_paths, str):
            image_paths = [image_paths]
            
        valid_images = []
        for p in image_paths:
            if os.path.exists(p):
                valid_images.append(Image.open(p))
                
        if not valid_images:
            return "Yellow Toy Car (Simulated - No Image)"
            
        print(f":: Asking Gemini about {len(valid_images)} images...")
        
        try:
            prompt = "Identify the main object. Return ONLY the name (e.g. 'Red Car')."
            response = self.model.generate_content([prompt] + valid_images)
            label = response.text.strip()
            print(f"   -> Gemini says: '{label}'")
            return label
            
        except Exception as e:
            print(f"   [Error] Gemini API failed: {e}")
            # 실패 시 모의 라벨 반환
            return "Yellow Toy Car (Simulated - API Error)"