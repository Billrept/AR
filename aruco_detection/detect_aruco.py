import cv2
import numpy as np
from generate_aruco import ARUCO_DICT

def aruco_display(corners, ids, rejected, image):
    if len(corners) > 0:
        ids = ids.flatten()

        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            
            # Coordinates of each corner of each marker (if needed)
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            # Coordinates of center of marker (if needed)
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

            cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 2)
            print(f"Detected ArUco marker ID: {markerID}")

    return image

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    
    parameters = cv2.aruco.DetectorParameters()

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    if ids is not None:
        for i in range(len(ids)):
            marker_length = 0.1  # Marker size in meters (Change to acutal values)
            object_points = np.array([
                [-marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0]
            ], dtype=np.float32)

######################## Take rvec and tvec here for things and stuff ###############################################
            # rvec = rotation vector
            # tvec = Translation vector
            
            success, rvec, tvec = cv2.solvePnP(
                object_points, corners[i][0], matrix_coefficients, distortion_coefficients
            )

            if success:

                cv2.aruco.drawDetectedMarkers(frame, corners)
                cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)

    return frame


def detect_aruco(): 
    # Change according to wanted detection
    aruco_type = "DICT_7X7_1000"

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])

    aruco_params = cv2.aruco.DetectorParameters()

    # Calculate both of this again on another camera
    intrinsic_camera = np.array(((637.97833597, 0, 316.43616487), (0, 638.78431852, 238.96752943), (0, 0, 1)))
    distortion = np.array((-0.05768105, 0.5123163, -0.00516302, 0.00400248, -1.74148176))

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():

        ret, img = cap.read()

        h, w, _ = img.shape

        width = 1000
        height = int(width*(h/w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        corners, ids, rejected = detector.detectMarkers(img)

        # detected_markers = aruco_display(corners, ids, rejected, img)
        output = pose_estimation(img, ARUCO_DICT[aruco_type], intrinsic_camera, distortion)

        cv2.imshow("Press q to exit", output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    detect_aruco()