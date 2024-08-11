# horizontally symmetric

import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('horizontal.png')

if img is None :
    print('Could not open or find the image')

else:   
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_symmetry = img.copy()
    cv2.line(img_symmetry, (0, gray.shape[0]//2), (gray.shape[1], gray.shape[0]//2), (0, 255, 0), 2)

# save the image
cv2.imwrite('horizontal_symmetry.png', img_symmetry)
# display the image
plt.imshow(cv2.cvtColor(img_symmetry, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

