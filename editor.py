import cv2
import os
import torch
from ultralytics import YOLO
from torchvision.ops import nms

# Paths
input_folder = 'input_images'
output_folder = 'output_images'
os.makedirs(output_folder, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Process each image
for img_name in os.listdir(input_folder):
    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(input_folder, img_name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape

        # Inference
        results = model(image)[0]

        # Extract person boxes and confidence scores
        boxes = []
        scores = []
        for det in results.boxes:
            if int(det.cls[0]) == 0:  # class 0 = person
                x1, y1, x2, y2 = map(float, det.xyxy[0])
                conf = float(det.conf[0])
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)

        # Apply NMS
        if boxes:
            boxes_tensor = torch.tensor(boxes)
            scores_tensor = torch.tensor(scores)
            keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold=0.5)

            # Label each retained detection
            for idx, i in enumerate(keep_indices, start=1):
                x1, y1, x2, y2 = map(int, boxes[i])
                label = f'{idx}'

                # Font and text setup
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.5
                thickness = 2
                text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_width, text_height = text_size

                # Center the label horizontally above the bounding box
                center_x = (x1 + x2) // 2
                text_x = center_x - (text_width // 2)
                text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10

                # Draw black background
                top_left = (text_x, text_y - text_height - baseline)
                bottom_right = (text_x + text_width, text_y + baseline)
                cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), cv2.FILLED)

                # Draw red label
                cv2.putText(image, label, (text_x, text_y), font,
                            font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        # Save image
        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, image)
        print(f'Labeled: {img_name}')
