from ultralytics import YOLO
import cv2

# 学習したbest.ptを使って引数に渡されたframe画像からドアオブジェクトを検知


def detect_door(frame):
    model = YOLO("door-detection-model.pt")
    results = model(frame)[0]

    door_class_id = [k for k, v in model.names.items() if v == 'door'][0]
    door_boxes = []

    for box in results.boxes:
        if int(box.cls) == door_class_id:
            door_boxes.append(box)

    for box in door_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame