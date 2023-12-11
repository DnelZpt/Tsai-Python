"""
Basics functions for camera calibration using Tsai's method

CV - MIE UTP
2023
"""

import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def get_projection_matrix(A):
    """
    This function takes in a matrix A and computes its singular value decomposition (SVD) using the numpy.linalg.svd function. It then extracts the right singular vectors and performs a projection
    * onto the last vector by setting all other elements to zero except for the last element, which is set to 1. Finally, the method returns the projected matrix.

    :param A: A matrix representing the input data.
    :return: The projected matrix using the Singular Value Decomposition (SVD) method.
    """
    U, s, VT = np.linalg.svd(A)
    V = VT.T
    mask = np.zeros((V.shape[0], 1))
    mask[-1] = 1
    projection = np.dot(V, mask)

    return projection


def get_A(data):
    """
    This function takes in a numpy array of data and computes the A matrix based on the given formula. The A matrix is then returned.

    :param data: a numpy array of shape (n_points, 5) containing the input data. Each row represents a data point with 5 values: [x, y, z, a, b].
    :return: A numpy array of shape (2 * n_points, 12) representing the A matrix.
    """
    n_points, _ = data.shape
    A_m = np.zeros((n_points * 2, 12))  # initialize A matrix
    first = 0
    sec = 2
    for points in data:
        spam = np.zeros((2, 12))

        spam[0, :3] = points[:3]
        spam[0, 3] = 1
        spam[0, 8:11] = points[:3] * -(points[3])
        spam[0, 11] = -points[3]

        spam[1, 4:7] = points[:3]
        spam[1, 7] = 1
        spam[1, 8:11] = points[:3] * -(points[4])
        spam[1, 11] = -points[4]

        A_m[first:sec, :] = spam

        first = sec
        sec += 2
    return A_m


def get_projection_matrix_hardcode(A):
    """
    Compute the projection matrix using hardcoded parameters.

    :param A: The input matrix A.
    :return: The projection matrix.
    """

    AtA = np.dot(A.T, A)
    eigenvalue, eigenvector = LA.eig(AtA)
    pick = np.argmin(eigenvalue)
    mask = np.zeros((eigenvector.shape[0], 1))
    mask[pick] = 1
    projection = np.dot(eigenvector, mask)

    return projection


def get_parameters(projection_matrix):
    """
    :param projection_matrix: A 3x4 projection matrix represented as a numpy array.

    :return: A tuple of three numpy arrays:
        - intrinsic matrix (3x3) after normalization
        - rotation matrix (3x3)
        - translation vector (3x1)

    """
    Q = projection_matrix[:, :3]
    b = projection_matrix[:, -1]
    b = b.reshape(3, 1)

    translation = np.dot((LA.inv(-Q)), b)

    Qinv = np.linalg.inv(Q)
    Rt, Kinv = np.linalg.qr(Qinv)

    intrinsic = np.linalg.inv(Kinv)
    intrinsic_n = intrinsic / intrinsic[2, 2]
    rotation = np.transpose(Rt)

    return abs(intrinsic_n), rotation, translation


def reprojection_error(projection_matrix, rotation, translation, data):
    """
        This method calculates the reprojection error between the projected image points and the given data points.

        :param projection_matrix: The projection matrix.
        :param rotation: The rotation matrix.
        :param translation: The translation vector.
        :param data: The data points.

        :return: The mean reprojection error.

    """
    Q = projection_matrix[:, :3]
    Qinv = np.linalg.inv(Q)
    Rt, Kinv = np.linalg.qr(Qinv)
    intrinsic = np.linalg.inv(Kinv)

    f = np.dot(intrinsic, rotation)
    s = np.dot(-Q, translation)
    pcon = np.hstack((f, s))

    wp = data[:, :4].copy()
    wp[:, -1] = 1
    image_points = np.dot(pcon, wp.T).T
    image_points = image_points / image_points[:, 2].reshape(-1, 1)
    print("\nProjected points are:")
    print(image_points[:, :2])
    error = abs(image_points[:, :2] - data[:, 3:])
    return np.mean(error, axis=0)


def plot_extrinsics(rotation, translation):
    """
    This function plots the extrinsics of a camera using Matplotlib.

    :param rotation: A list of rotation matrices representing the camera rotation in each frame.
    :param translation: A list of translation vectors representing the camera translation in each frame.
    :return: None

    """
    print("\nVisualizing extrinsics now!, computing camera location...")
    fig = plt.figure()
    ax = Axes3D(fig)
    count = 1
    for rotation, translation in zip(rotation, translation):
        mag = 50

        corners = np.array([
            [-mag, mag, 0],
            [mag, mag, 0],
            [mag, -mag, 0],
            [-mag, -mag, 0],
            [-mag, mag, 0],
            [0, 0, mag],
            [-mag, -mag, 0],
            [0, 0, mag],
            [mag, mag, 0],
            [0, 0, mag],
            [mag, -mag, 0],
            [0, 0, mag],
            [0, 0, 0],
            [mag, 0, 0],
            [0, 0, 0],
            [0, -mag, 0],
            [0, 0, 0],
        ])

        # This give coordinates of camera in world coordinate
        camera_points = np.dot(corners, rotation) + translation.reshape(1,
                                                                        3)
        camera_points = np.vstack((camera_points, np.array([0, 0, 0])))
        print("\nCamera corners in world coordinates are:\n", camera_points)

        x = camera_points[:, 0]
        y = camera_points[:, 1]
        z = camera_points[:, 2]

        ax.scatter(x, y, z, color='black', depthshade=False, s=6)
        ax.text(camera_points[5, 0], camera_points[5, 1], camera_points[5, 2], str(count), fontsize=20,
                color='darkblue')
        ax.set_xlim([-10, 500])
        ax.set_ylim([-10, 500])
        ax.set_zlim([-10, 500])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('equal', 'box')
        ax.plot(x, y, z, color='green')
        ax.invert_yaxis()
        count += 1


if __name__ == '__main__':
    rotation_matrices, translation_vectors = [], []
    data_imgs = np.loadtxt('data/img_01.txt')
    A_m = get_A(data_imgs)

    projection_matrix = get_projection_matrix(A_m)
    projection_matrix = projection_matrix.reshape(3, 4)
    print("\nProjection matrix :\n", projection_matrix)

    intrinsic, rotation, translation = get_parameters(projection_matrix)
    rotation_matrices.append(rotation)
    translation_vectors.append(translation)

    print("\nIntrinsics:\n", intrinsic)  # K matrix
    print("\nRotation:\n", rotation)
    print("\nTranslation:\n", translation)

    error = reprojection_error(projection_matrix, rotation, translation, data_imgs)
    print("\nReprojection error is:", error)

    plot_extrinsics(rotation_matrices, translation_vectors)

    plt.show()
