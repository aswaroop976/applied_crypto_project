# tests/hash_check.py
def hamming_distance(h1, h2):
    """Compute Hamming distance between two PDQ bit vectors."""
    return sum(b1 != b2 for b1, b2 in zip(h1, h2))

def pdq_to_hex(hv):
    """Convert PDQ bit-vector (list of 0/1 ints) into a 64-char hex string."""
    bitstring = ''.join(str(b) for b in hv)          # '010110...'
    as_int = int(bitstring, 2)                       # binary → int
    hex_str = f"{as_int:064x}"                       # 64 hex chars (256 bits)
    return hex_str
import cv2, pdqhash, imagehash
from PIL import Image
img = cv2.imread('/home/arpan/Documents/crypto/project/jpg/image_00001.jpg')
im2 = cv2.imread('/home/arpan/Documents/crypto/project/output_with_rectangle.jpg')
#img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
hv, q = pdqhash.compute(img)
hv2, q2 = pdqhash.compute(im2)
print("PDQ hash vector len:", len(hv), "quality:", q)
print("PDQ hash:", str(hv))
print("PDQ hash:", str(hv2))
print("PDQ hash(hex):", pdq_to_hex(hv))
print("PDQ hash(hex):", pdq_to_hex(hv2))
print("PDQ hash distance:", hamming_distance(hv, hv2))
p = imagehash.phash(Image.open('/home/arpan/Documents/crypto/project/src/images_in/Stevens_Alfred_Jeune.jpg'))
p2 = imagehash.phash(Image.open('/home/arpan/Documents/crypto/project/output_with_rectangle.jpg'))
print("pHash:", str(p))

print("pHash:", str(p2))
print(f"type of phash hash:{type(p)}")
