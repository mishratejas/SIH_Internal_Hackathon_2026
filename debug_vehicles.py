"""
Debug victim/vehicle detection
"""
import cv2
from ultralytics import YOLO
import torch

print("="*70)
print("VEHICLE & PEOPLE DETECTION DEBUG")
print("="*70)

device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt")

img = cv2.imread("drones-10-00015-g001.png")
if img is None:
    print("Image not found")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"\nImage shape: {img_rgb.shape}")

# Test different confidence thresholds
for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
    print(f"\n[conf={conf}]")
    results = model.predict(
        source=img_rgb,
        conf=conf,
        device=device,
        verbose=False
    )
    
    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None:
            print(f"  Total detections: {len(result.boxes)}")
            
            # Count by class
            cars = 0
            people = 0
            motorcycles = 0
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:
                    people += 1
                elif class_id == 2:  # car
                    cars += 1
                elif class_id == 3:  # motorcycle
                    motorcycles += 1
                
                # Print first 5 detections
                if len(result.boxes) <= 5 or (cars + people + motorcycles) <= 5:
                    conf_score = float(box.conf[0])
                    print(f"    class={class_id}, conf={conf_score:.3f}")
            
            print(f"  👤 People: {people}")
            print(f"  🚗 Cars: {cars}")
            print(f"  🏍️ Motorcycles: {motorcycles}")
