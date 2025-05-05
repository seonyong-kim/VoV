# cmd에서
# pip install easyocr opencv-python Pillow matplotlib
# pip install easyocr

import easyocr
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# EasyOCR 리더 객체 생성 (한글 + 영어)
reader = easyocr.Reader(['ko', 'en'], gpu=False)

# OCR 실행: test.jpg에서 텍스트 추출
results = reader.readtext('test.jpg')

# 이미지 로딩
image = cv2.imread('test.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image)

# 폰트 설정 (윈도우용 경로 예시)
font_path = "C:/Windows/Fonts/malgun.ttf"  # '맑은 고딕' 폰트
try:
    font = ImageFont.truetype(font_path, 20)  # 글자 크기 조절
except IOError:
    font = ImageFont.load_default()
    print("⚠️ 지정한 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

# 그림 그리기 객체
draw = ImageDraw.Draw(image_pil)
np.random.seed(42)

# OCR 결과 처리
for (bbox, text, _) in results:
    # bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    x_coords = [point[0] for point in bbox]
    y_coords = [point[1] for point in bbox]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    color = tuple(np.random.randint(0, 256, 3).tolist())

    # 사각형 및 텍스트 출력
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline=color, width=2)
    draw.text((x_min, y_min - 25), text, font=font, fill=color)

# 결과 이미지 출력
plt.figure(figsize=(10, 10))
plt.imshow(np.asarray(image_pil))
plt.axis('off')
plt.show()

# 텍스트 콘솔 출력
print("\n[인식된 텍스트 목록]")
for (_, text, _) in results:
    print(f"- {text}")
