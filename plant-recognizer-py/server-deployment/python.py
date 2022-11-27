import urllib.request
import numpy as np
import json
import sys
import cv2

url = sys.argv[1]

url_response = urllib.request.urlopen(url)
image = cv2.imdecode(np.array(bytearray(url_response.read()), dtype=np.uint8), -1)

cv2.imwrite("image.png", image)

output = input

print(output)

sys.stdout.flush()