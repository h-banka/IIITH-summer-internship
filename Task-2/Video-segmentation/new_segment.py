from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8m-seg.pt")

input_folder = "frames"
output_folder = "segmented_frames"
os.makedirs(output_folder, exist_ok=True)

for filename in sorted(os.listdir(input_folder)):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)
        results = model(image_path)

        for result in results:
            img = result.plot()
            output_path = os.path.join(output_folder, f"{filename.split('.')[0]}_pred.jpg")
            cv2.imwrite(output_path, img)

print("Segmentation complete. Segmented frames saved to 'segmented_frames/'")
