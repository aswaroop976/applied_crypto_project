# tests/hash_check.py
import cv2, pdqhash, imagehash
from PIL import Image
img = cv2.imread('/home/arpan/Documents/crypto/project/src/images_in/Stevens_Alfred_Jeune.jpg')
print(type(img))
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hv, q = pdqhash.compute(img)
print("PDQ hash vector len:", len(hv), "quality:", q)
p = imagehash.phash(Image.open('/home/arpan/Documents/crypto/project/src/images_in/Stevens_Alfred_Jeune.jpg'))
print("pHash:", str(p))
print(f"type of phash hash:{type(p)}")
