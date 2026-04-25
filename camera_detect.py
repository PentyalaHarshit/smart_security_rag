import cv2
from ultralytics import YOLO
from datetime import datetime
import os

EVENT_FILE = "data/events.txt"
os.makedirs("data", exist_ok=True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_seen = {}

if not cap.isOpened():
    print("Camera not opened. Try cv2.VideoCapture(1)")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45)
    annotated = results[0].plot()

    detected_objects = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        detected_objects.append(name)

    for obj in set(detected_objects):
        now = datetime.now()

        if obj not in last_seen or (now - last_seen[obj]).seconds > 10:
            last_seen[obj] = now

            event = f"{now.strftime('%Y-%m-%d %H:%M:%S')} - Detected {obj}"
            print(event)

            with open(EVENT_FILE, "a", encoding="utf-8") as f:
                f.write(event + "\n")

    cv2.imshow("AI Smart Security Camera", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()