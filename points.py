import numpy as np
import cv2

path = 'shoots/'


def load_image(file_name):
    image = cv2.imread(path + file_name)
    return image


if __name__ == '__main__':
    image_01 = load_image(path + '1.jpg')

