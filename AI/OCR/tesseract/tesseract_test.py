import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'Tesseract 설치 경로\tesseract.exe'
img = Image.open('test2.jpg') 
text = pytesseract.image_to_string(img, lang='kor')

print(text)
