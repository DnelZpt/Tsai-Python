"""
Main module for Tsai's application in camera calibration


Daniela Buitrago, Kevin Ortega, Daniel Zapata
MIE - UTP
2023
"""

import numpy as np
import basics
from scipy.linalg import svd

path = 'data/'
corners_files = ['img_01.txt', 'img_02.txt', 'img_03.txt', 'img_04.txt', 'img_05.txt']

for file in corners_files:
    data = np.loadtxt(path + file)
    print(file + ':')
    print(data)

    A_p = basics.get_A(data)


