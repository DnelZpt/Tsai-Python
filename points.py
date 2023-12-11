"""
Module to get corners points from the Tsai instrument

Computer Vision - MIE UTP
2023
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def load_image(file_path):
    """
    Load an image in gray scale from the specified file path.

    :param file_path: A string representing the path to the image file.
    :return: The loaded image as a grayscale image.
    """
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image


def get_corner_points(image):
    """
    Get the corner points of an image using the goodFeaturesToTrack method.

    :param image: The input image.
    :return: The corner points of the image as a numpy array.
    """
    corners = cv2.goodFeaturesToTrack(image, 10, 0.01, 10)
    return np.int64(corners)


def show_corners(image, corners):
    """
    Visualizes corners on an image.

    :param image: A numpy array representing the image.
    :param corners: A list of corner coordinates.
    :return: None

    Example usage:
        >>> image = cv2.imread('image.jpg')
        >>> corners = [[10, 20], [30, 40], [50, 60]]
        >>> show_corners(image, corners)
    """
    count = 1  # Just a counter to identify every corner
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 10, 255, -1)
        cv2.putText(image, str(count), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 255, 0), 2, cv2.LINE_AA)
        count += 1

    plt.imshow(image, cmap='gray'), plt.show()


def corners_report(corners):
    """
    Prints a report of the corners found in the input array. It displays the number of corners
    and prints each corner's x and y coordinates in a tabular format.

    :param corners: A NumPy array containing the corners coordinates.
    :return: None
    """

    num_corners = corners.shape[0]
    print(f"Number of corners: {num_corners}")
    print('Corners found:')
    print('ID     x     y')
    print('---------------')
    count = 1  # Just a counter to identify every corner
    for i in corners:
        x, y = i.ravel()
        print('%2d  %4d  %4d' % (count, x, y))
        count += 1


if __name__ == '__main__':
    path = 'shoots/'
    image_01 = load_image(path + '1.jpg')
    image_02 = load_image(path + '2.jpg')
    image_03 = load_image(path + '3.jpg')
    image_04 = load_image(path + '4.jpg')
    image_05 = load_image(path + '5.jpg')

    print('Img 01:')
    plt.figure(1)
    plt.title('Img 01')
    corners_01 = get_corner_points(image_01)
    show_corners(image_01, corners_01)
    corners_report(corners_01)
    print('\n')

    print('Img 02:')
    plt.figure(2)
    plt.title('Img 02')
    corners_02 = get_corner_points(image_02)
    show_corners(image_02, corners_02)
    corners_report(corners_02)
    print('\n')

    print('Img 03:')
    plt.figure(3)
    plt.title('Img 03')
    corners_03 = get_corner_points(image_03)
    show_corners(image_03, corners_03)
    corners_report(corners_03)
    print('\n')

    print('Img 04:')
    plt.figure(4)
    plt.title('Img 04')
    corners_04 = get_corner_points(image_04)
    show_corners(image_04, corners_04)
    corners_report(corners_04)
    print('\n')

    print('Img 05:')
    plt.figure(5)
    plt.title('Img 05')
    corners_05 = get_corner_points(image_05)
    show_corners(image_05, corners_05)
    corners_report(corners_05)

    # cv2.namedWindow('Tsai', cv2.WINDOW_NORMAL)

    # Change the size for renamed window
    # cv2.resizeWindow('Tsai', 1600, 900)

    # cv2.imshow('Tsai', image_01)
    # cv2.waitKey(0)  # Close the window with 'q' key
    # cv2.destroyAllWindows()
