import cv2
import cairosvg
import matplotlib.pyplot as plt
import numpy as np

def plot_horizontal_symmetry(img, img_no):
    if img is None :
        print('Could not open or find the image')

    else:   
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_symmetry = img.copy()
        cv2.line(img_symmetry, (0, gray.shape[0]//2), (gray.shape[1], gray.shape[0]//2), (0, 255, 0), 2)

    cv2.imwrite('./assets/horizontal/' + img_no + '.png', img_symmetry)


def plot_vertical_symmetry(img, img_no):
    if img is None:
        print('Could not open or find the image')

    else: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flip_vertically = cv2.flip(gray, 1)
        img_symmetry = img.copy()
        cv2.line(img_symmetry, (gray.shape[1]//2, 0), (gray.shape[1]//2, gray.shape[0]), (0, 255, 0), 2) 

    cv2.imwrite('./assets/vertical/' + img_no + '.png', img_symmetry)


def plot_diagonal_symmetry(img, img_no):
    if img is None:
        print('Could not open or find the image')

    else: 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flip_vertically = cv2.flip(gray, 1)
        img_symmetry = img.copy()
        cv2.line(img_symmetry, (0, 0), (gray.shape[1], gray.shape[0]), (0, 255, 0), 2) 

    cv2.imwrite('./assets/diagonal/' + img_no + '.png', img_symmetry)


def plot_symmetry():
    for i in range(0,3):
        i = str(i)
        file_path = './assets/images/regularized_plot_' + i + '.svg'
        cairosvg.svg2png(url=file_path, write_to='./temp_image.png')
        img = cv2.imread('./temp_image.png')
        # img = cv2.imread('./assets/images/regularized_plot_'+ i +'.svg')
        plot_horizontal_symmetry(img, i)
        plot_vertical_symmetry(img, i)
        plot_diagonal_symmetry(img, i)

plot_symmetry()
