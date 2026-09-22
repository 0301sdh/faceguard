from PIL import Image
import numpy as np

img = Image.open("data/person_a_1.jpg")

print("이미지 크기 (width, height):", img.size)
print("이미지 모드 (색상 채널):", img.mode)   # RGB면 3채널

img_array = np.array(img)
print("배열 shape:", img_array.shape)         # (height, width, channel)
print("배열 dtype:", img_array.dtype)         # 보통 uint8 (0~255)