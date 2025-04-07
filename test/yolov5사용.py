import torch
import cv2
import numpy as np

# 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ 좌우 반전
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # YOLO 추론
    results = model(img)

    # 결과 렌더링
    results.render()

    # RGB → BGR
    output = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Object Detection (Flipped)', output)

    # ESC 키로 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
s