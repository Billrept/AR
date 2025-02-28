import cv2
import numpy as np

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])
            elif parts[0] == 'f':
                idxs = []
                for vtx in parts[1:]:
                    v = vtx.split('/')[0]
                    idxs.append(int(v) - 1) 
                faces.append(idxs)
    return {
        'vertices': np.array(vertices, dtype=np.float32),
        'faces': faces
    }

def compute_scale_factor(obj_data, desired_size_meters=0.2):
    verts = obj_data['vertices']
    min_xyz = verts.min(axis=0)
    max_xyz = verts.max(axis=0)
    bbox = max_xyz - min_xyz
    current_size = np.max(bbox)
    scale_factor = desired_size_meters / current_size
    return scale_factor

def scale_obj(obj_data, scale_factor):
    obj_data['vertices'] *= scale_factor

def project_obj_solid(frame, rvec, tvec, obj_data, camera_matrix, dist_coeffs):
    R, _ = cv2.Rodrigues(rvec)

    vertices_obj = obj_data['vertices']
    faces = obj_data['faces']

    vertices_cam = []
    for pt in vertices_obj:
        pt3d = pt.reshape(3, 1)
        pt_cam = R @ pt3d + tvec
        vertices_cam.append(pt_cam.flatten())
    vertices_cam = np.array(vertices_cam)

    projected_pts_2d, _ = cv2.projectPoints(vertices_obj, rvec, tvec, camera_matrix, dist_coeffs)
    projected_pts_2d = projected_pts_2d.reshape(-1, 2).astype(int)

    face_depths = []
    for f_idx, face in enumerate(faces):
        cam_z = vertices_cam[face, 2]
        avg_z = np.mean(cam_z)
        face_depths.append((f_idx, avg_z))
    face_depths.sort(key=lambda x: x[1], reverse=True)

    for (face_idx, _) in face_depths:
        idxs = faces[face_idx]
        face_2d = projected_pts_2d[idxs].reshape(-1, 1, 2)

        cv2.fillConvexPoly(frame, face_2d, (180, 180, 180), lineType=cv2.LINE_AA)
        cv2.polylines(frame, [face_2d], True, (0, 0, 0), 2, cv2.LINE_AA)

    return frame

def main():
    #Load da OBJ model
    obj_data = load_obj("/Users/phacharakimpha/")

    scale_factor = compute_scale_factor(obj_data, 0.15)
    scale_obj(obj_data, scale_factor)

    #Mine (estimataion) maacbook 1080p intrinsic
    camera_matrix = np.array([
        [1500, 0,   960],
        [   0, 1500, 550],
        [   0,    0,   1]
    ], dtype=np.float32)
    dist_coeffs = np.array([-0.2, 0.05, 0.001, 0.001, -0.01], dtype=np.float32)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
    aruco_params = cv2.aruco.DetectorParameters()

    marker_size = 0.1
    
    marker_corners_3d = np.array([
        [-marker_size/2,  marker_size/2,  0],
        [ marker_size/2,  marker_size/2,  0],
        [ marker_size/2, -marker_size/2,  0],
        [-marker_size/2, -marker_size/2,  0]
    ], dtype=np.float32)

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

                    frame = project_obj_solid(frame, rvec, tvec, obj_data, camera_matrix, dist_coeffs)

        cv2.imshow("AR OBJ Overlay (Depth-Sorted)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
