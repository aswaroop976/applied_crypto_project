# tests/hash_check.py
import cv2, pdqhash, imagehash
from PIL import Image
img = cv2.imread('/home/arpan/Documents/crypto/project/adversarial-detection-avoidance-attacks/datasets/n01440764_18.JPEG')
print(type(img))
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hv, q = pdqhash.compute(img)
print("PDQ hash vector len:", len(hv), "quality:", q)
p = imagehash.phash(Image.open('/home/arpan/Documents/crypto/project/adversarial-detection-avoidance-attacks/datasets/n01440764_18.JPEG'))
print("pHash:", str(p))
