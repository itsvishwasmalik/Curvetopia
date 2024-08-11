import cv2
import matplotlib.pyplot as plt
import numpy as np

# reading the image
img = cv2.imread('../regularize_curves/regularized_plot.png')

# converting the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# applying the thresholding
_, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# finding the contours
contour, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
i=0

# list for storing the names of all the shapes

for cnt in contour:
    if i==0:
        i+=1
        continue

    approx = cv2.approxPolyDP(
        cnt, 0.01*cv2.arcLength(cnt, True), True)
    
    #drawcontours function
    cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)

    # Finding the center point of the shape
    M = cv2.moments(cnt)

    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00'])
        y = int(M['m01']/M['m00'])

    # putting the names of the shapes at the center of the shapes

    if len(approx) == 3:
        cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif len(approx) == 4:
        cv2.putText(img, 'Rectangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif len(approx) == 5:
        cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    elif len(approx) == 6:
        cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cv2.putText(img, 'Circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
# displaying the image
cv2.imshow('shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()