"""
Main module for Tsai's application in camera calibration


Daniela Buitrago, Kevin Ortega, Daniel Zapata
MIE - UTP
2023
"""

import numpy as np
import basics
import matplotlib.pyplot as plt

path = 'data/'
corners_files = ['img_01.txt', 'img_02.txt', 'img_03.txt', 'img_04.txt', 'img_05.txt']

rotation_matrices, translation_vectors = [], []

for file in corners_files:
    data_imgs = np.loadtxt(path + file)
    print(file + ':')
    print(data_imgs)

    A_m = basics.get_A(data_imgs)

    projection_matrix = basics.get_projection_matrix(A_m)
    projection_matrix = projection_matrix.reshape(3, 4)
    print("\nProjection matrix :\n", projection_matrix)

    intrinsic, rotation, translation = basics.get_parameters(projection_matrix)
    rotation_matrices.append(rotation)
    translation_vectors.append(translation)

    print("\nIntrinsics:\n", intrinsic)  # K matrix
    print("\nRotation:\n", rotation)
    print("\nTranslation:\n", translation)

    error = basics.reprojection_error(projection_matrix, rotation, translation, data_imgs)
    print("\nReprojection error is:", error)

basics.plot_extrinsics(rotation_matrices, translation_vectors)
plt.show()
