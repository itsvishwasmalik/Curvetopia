# Vertical symmetry detection 
import cv2
import matplotlib.pyplot as plt
import numpy as np


# Load the image
img = cv2.imread('butt.png')

if img is None:
    print('Could not open or find the image')
    exit(0)

else: 
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Flip the image vertically
    flip_vertically = cv2.flip(gray, 1)

    # draw the vertical symmetry line
    img_symmetry = img.copy()
    cv2.line(img_symmetry, (gray.shape[1]//2, 0), (gray.shape[1]//2, gray.shape[0]), (0, 255, 0), 2) 

# Save the image
cv2.imwrite('vertical_symmetry.png', img_symmetry)
# Display the image
plt.imshow(cv2.cvtColor(img_symmetry, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
