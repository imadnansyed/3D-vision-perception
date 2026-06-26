from pathlib import Path
import cv2

class ImageLoader:
    def __init__(self, image_dir:str):
        self.dataset_path = Path(image_dir)
    
    def load_images(self):
        
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_paths = sorted([p for p in self.dataset_path.iterdir() if p.suffix.lower() in valid_extensions])
        
        images = []
        
        for idx, image_path in enumerate(image_paths):
            
            image = cv2.imread(str(image_path))
            
            if image is None:
                print(f"Cannot load image : {image_path}")
                continue
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            images.append({
                "id": idx,
                "name": image_path.name,
                "image": image,
                "gray": gray
            })
        
        return images
    
    
            
                