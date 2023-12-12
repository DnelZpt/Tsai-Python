# import points
import numpy as np
import cv2


def draw(img, img_pts, color=(0, 255, 0)):
    """
    Draw lines connecting the projected 3D object points on the image.

    :param img: The input image.
    :param img_pts: The input image points
    :param color: Color of the lines
    :return: The image with lines connecting the projected 3D object points.
    """

    img_pts = np.int64(img_pts).reshape(-1, 2)
    for i in range(len(img_pts)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[(i + 1) % len(img_pts)]), color, 3)
    return img


if __name__ == '__main__':
    """image = points.load_image('test.jpg')
    corners = points.get_corner_points(image)
    points.show_corners(image, corners)
    points.corners_report(corners)"""

    with np.load('calibration_parameters.npz') as X:
        intrinsic_matrices = X['intrinsic_matrices']
        rotation_matrices = X['rotation_matrices']
        translation_vectors = X['translation_vectors']

    image_points_2d = np.array(
        [[435, 424], [1053, 769], [560, 788], [1140, 1048], [463, 1054], [663, 977], [1177, 398], [652, 300],
         [948, 960],
         [552, 423]])

    object_points_3d = np.array([
        [126.5, 0, 188.6],
        [0, 96.5, 98.6],
        [96.5, 0, 98.6],
        [0, 126.5, 38.6],
        [126.5, 0, 38.6],
        [66.5, 0, 38.6],
        [0, 126.5, 188.6],
        [66.5, 0, 218.6],
        [0, 66.5, 38.6],
        [96.5, 0, 188.6]
    ])

    image = cv2.imread('test.jpg')

    idx = 3  # Parameters set to use index
    mtx = intrinsic_matrices[idx]
    rmat = rotation_matrices[idx]
    tvec = translation_vectors[idx]

    img_pts_projected, _ = cv2.projectPoints(object_points_3d, rmat, tvec, mtx, None)
    img_pts_projected = np.int64(img_pts_projected.reshape((10, 2)))

    image_with_object = draw(image, image_points_2d)
    image_with_proj_lines = draw(image_with_object, img_pts_projected, color=(0, 0, 255))

    cv2.namedWindow('Reality Augmented', cv2.WINDOW_NORMAL)

    # Change the size for renamed window
    cv2.resizeWindow('Reality Augmented', 1600, 900)

    cv2.imshow('Reality Augmented', image_with_proj_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
