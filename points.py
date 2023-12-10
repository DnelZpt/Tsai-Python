import numpy as np
import cv2


def load_image(file_path):
    """
    Load an image from the specified file path.

    :param file_path: A string representing the path to the image file.
    :return: The loaded image as a grayscale image.
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image


if __name__ == '__main__':
    path = 'shoots/'
    image_01 = load_image(path + '1.jpg')
    cv2.namedWindow('Tsai', cv2.WINDOW_NORMAL)

    # Change the size for renamed window
    cv2.resizeWindow('Tsai', 1600, 900)

    cv2.imshow('Tsai', image_01)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
