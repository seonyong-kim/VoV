import torch
import cv2
import numpy as np
import pyttsx3

# TTS 엔진 초기화
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # 말하는 속도 조절

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 카메라 열기
cap = cv2.VideoCapture(0)

spoken_labels = set()  # 이미 말한 라벨 저장용

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 좌우 반전
    frame = cv2.flip(frame, 1)

    # BGR → RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 추론
    results = model(img)
    results.render()

    # 인식된 객체 정보 가져오기
    labels = results.names
    preds = results.pred[0]  # 예측 결과 (x1, y1, x2, y2, conf, class)

    for *box, conf, cls in preds:
        label = labels[int(cls)]
        if label not in spoken_labels:
            sentence = f"{label} 감지되었습니다"
            engine.say(sentence)
            engine.runAndWait()
            spoken_labels.add(label)

    # 출력
    output = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
    cv2.imshow('YOLOv5 Object Detection + TTS', output)

    # ESC 키로 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
