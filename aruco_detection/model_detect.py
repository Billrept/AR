import cv2
import numpy as np

camera_matrix = np.array([
    [1500,   0.        , 960],
    [  0.        , 1500, 550],
    [  0.        ,   0.        ,   1.0]
], dtype=np.float32)

dist_coeffs = np.array([-0.2, 0.05, 0.001, 0.001, -0.01], dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
aruco_params = cv2.aruco.DetectorParameters()

marker_size = 0.10
marker_corners_3d = np.array([
    [-marker_size/2, -marker_size/2,  0],
    [ marker_size/2, -marker_size/2,  0],
    [ marker_size/2,  marker_size/2,  0],
    [-marker_size/2,  marker_size/2,  0]
], dtype=np.float32)

cube_size = 0.12
cube_points_3d = np.float32([
    [0, 0, 0],
    [cube_size, 0, 0],
    [cube_size, cube_size, 0],
    [0, cube_size, 0],
    [0, 0, -cube_size],
    [cube_size, 0, -cube_size],
    [cube_size, cube_size, -cube_size],
    [0, cube_size, -cube_size]
])

faces = [
    (0, 1, 2, 3),  # bottom
    (4, 5, 6, 7),  # top
    (0, 1, 5, 4),  # side 1
    (1, 2, 6, 5),  # side 2
    (2, 3, 7, 6),  # side 3
    (3, 0, 4, 7)   # side 4
]

face_colors = [
    (255, 0, 0),    # bottom: blue
    (0, 255, 0),    # top: green
    (0, 0, 255),    # side 1: red
    (255, 255, 0),  # side 2: cyan
    (255, 0, 255),  # side 3: magenta
    (0, 255, 255)   # side 4: yellow
]


def project_solid_cube(frame, rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)

    cube_cam_space = []
    for pt in cube_points_3d:
        pt3d = np.array([[pt[0]], [pt[1]], [pt[2]]], dtype=np.float32)
        pt_cam = R @ pt3d + tvec
        cube_cam_space.append(pt_cam.flatten())
    cube_cam_space = np.array(cube_cam_space)

    projected_pts_2d, _ = cv2.projectPoints(cube_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    projected_pts_2d = projected_pts_2d.reshape(-1, 2).astype(int)

    face_depths = []
    for fi, face in enumerate(faces):
        face_cam_pts = cube_cam_space[list(face)]
        avg_z = np.mean(face_cam_pts[:, 2])
        face_depths.append((fi, avg_z))

    face_depths.sort(key=lambda x: x[1], reverse=True)

    for (face_idx, _) in face_depths:
        pts_idx = faces[face_idx]
        color = face_colors[face_idx]

        pts_2d = projected_pts_2d[list(pts_idx)]
        pts_2d = pts_2d.reshape(-1, 1, 2)

        cv2.fillConvexPoly(frame, pts_2d, color, lineType=cv2.LINE_AA)
        cv2.polylines(frame, [pts_2d], True, (0,0,0), 2, cv2.LINE_AA)

    return frame


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            for marker_corner_2d, marker_id in zip(corners, ids):
                marker_corner_2d = marker_corner_2d[0]

                success, rvec, tvec = cv2.solvePnP(
                    marker_corners_3d,
                    marker_corner_2d,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )

                if success:
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
                    frame = project_solid_cube(frame, rvec, tvec)

        cv2.imshow("Aruco Solid Cube (Depth-Sorted)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
