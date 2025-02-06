import numpy as np
import cv2
import matplotlib.pyplot as plt

# Larger grid size = slower detection but less likely to have errors
# Larger tag size = larger physical size of marker

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
}

def generate_aruco():
    aruco_type = "DICT_4X4_250"

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    marker_size = 250
    id = 1

    marker_image = cv2.aruco.generateImageMarker(aruco_dict, id, marker_size)
    marker_name = aruco_type + "_" + str(id) + ".png"

    cv2.imwrite(f'./aruco_detection/aruco_markers/' + marker_name, marker_image)
    plt.imshow(marker_image, cmap='gray')
    plt.title(f'{aruco_type}')
    plt.show()


