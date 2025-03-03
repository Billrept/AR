import cv2
import numpy as np

from generate_aruco import ARUCO_DICT

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def line_intersection(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-10:
        return None
    
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    return (int(x), int(y))

def detect_aruco_markers(frame=None, camera_index=0, marker_length=0.049,
                        intrinsic_camera=None, distortion=None, marker_type="DICT_7X7_1000",
                        debug_display=False):
    """
    Detect ArUco markers and calculate central pose.
    
    Args:
        frame: Optional pre-loaded frame to use (if None, captures from camera)
        camera_index: Camera device index
        marker_length: Real marker size in meters
        intrinsic_camera: Camera intrinsic matrix (3x3)
        distortion: Camera distortion coefficients
        marker_type: ArUco dictionary type
        debug_display: Whether to show debug visualization window
        
    Returns:
        dict: Contains:
            - central_rvec: Rotation vector of central pose
            - central_tvec: Translation vector of central pose
            - central_distance: Distance to central pose in meters
            - central_point: 2D coordinates of central point in image
            - markers: List of detected marker data (position, orientation)
            - frame: Processed frame with visualizations (if debug_display is True)
            - success: Whether markers were successfully detected
    """
    # Default camera parameters if not provided
    if intrinsic_camera is None:
        intrinsic_camera = np.array([
        [
            957.1108018720613,
            0.0,
            556.0882651177826
        ],
        [
            0.0,
            951.9753671508217,
            286.42509589693657
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])
    
    if distortion is None:
        distortion = np.array(        [
            -0.25856927733393603,
            1.8456432127404514,
            -0.021219826734632862,
            -0.024902070756342175,
            -3.808238876719984
        ])
    
    aruco_dict_type = ARUCO_DICT.get(marker_type, cv2.aruco.DICT_7X7_1000)
    
    # If no frame is provided, capture from camera
    if frame is None:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            return {"success": False, "error": "Could not open camera"}

        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {"success": False, "error": "Failed to capture frame"}
            
        cap.release()
    
    h, w, _ = frame.shape
    width = 1280
    height = 720
    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    display_frame = frame.copy() if debug_display else None
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 19, 2
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_markers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > width*height*0.5:
            continue
            
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            rect = order_points(pts)
            
            width_marker = np.sqrt(((rect[1][0] - rect[0][0]) ** 2) + ((rect[1][1] - rect[0][1]) ** 2))
            height_marker = np.sqrt(((rect[3][0] - rect[0][0]) ** 2) + ((rect[3][1] - rect[0][1]) ** 2))

            if width_marker > 1e-6 and height_marker > 1e-6:
                aspect_ratio = width_marker / height_marker
                
                if 0.8 <= aspect_ratio <= 1.2:
                    dst_size = 100
                    dst_points = np.array([
                        [0, 0],
                        [dst_size-1, 0],
                        [dst_size-1, dst_size-1],
                        [0, dst_size-1]
                    ], dtype=np.float32)
                    
                    M = cv2.getPerspectiveTransform(rect, dst_points)
                    warped = cv2.warpPerspective(gray, M, (dst_size, dst_size))
                    
                    _, marker_binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                    border = 10
                    border_pixels = np.sum(marker_binary[:border, :]) + np.sum(marker_binary[-border:, :]) + \
                                  np.sum(marker_binary[:, :border]) + np.sum(marker_binary[:, -border:])
                    border_avg = border_pixels / (4 * border * dst_size)
                    
                    if border_avg > 200:
                        object_points = np.array([
                            [-marker_length/2, marker_length/2, 0],
                            [marker_length/2, marker_length/2, 0],
                            [marker_length/2, -marker_length/2, 0],
                            [-marker_length/2, -marker_length/2, 0]
                        ], dtype=np.float32)
                        
                        try:
                            success, rvec, tvec = cv2.solvePnP(
                                object_points, rect, intrinsic_camera, distortion
                            )
                            
                            if success:
                                center = np.mean(rect, axis=0).astype(int)
                                distance = np.linalg.norm(tvec) * 0.75
                                detected_markers.append({
                                    'rect': rect,
                                    'center': center,
                                    'rvec': rvec,
                                    'tvec': tvec,
                                    'distance': distance
                                })
                                
                                # Add visualization if debug mode is on
                                # if debug_display and display_frame is not None:
                                #     cv2.polylines(display_frame, [np.int32(rect)], True, (0, 255, 0), 2)
                                #     cv2.drawFrameAxes(display_frame, intrinsic_camera, 
                                #                      distortion, rvec, tvec, 0.05)
                                #     cv2.putText(display_frame, f"ArUco", 
                                #               (center[0]-20, center[1]-30),
                                #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                #     cv2.putText(display_frame, f"Dist: {distance:.3f}m",
                                #               (center[0]-30, center[1]-10),
                                #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                        except cv2.error:
                            pass

    # If no markers detected, return failure
    if not detected_markers:
        return {"success": False, "error": "No markers detected", 
                "frame": display_frame if debug_display else None}

    # Calculate central pose if markers are detected
    if len(detected_markers) >= 2:
        # Calculate average position for tvec and rvec (3D pose)
        tvecs = [marker['tvec'] for marker in detected_markers]
        central_tvec = np.mean(tvecs, axis=0)

        rvecs = [marker['rvec'] for marker in detected_markers]
        central_rvec = np.mean(rvecs, axis=0)
        
        # Calculate central distance
        central_distance = np.linalg.norm(central_tvec) * 0.75

        # Find central point based on number of markers
        if len(detected_markers) == 4:
            sorted_markers = sorted(detected_markers, key=lambda m: m['center'][0])
            
            top_left_idx = np.argmin([m['center'][0] + m['center'][1] for m in sorted_markers])
            bottom_right_idx = np.argmax([m['center'][0] + m['center'][1] for m in sorted_markers])
            
            remaining = [i for i in range(4) if i != top_left_idx and i != bottom_right_idx]
            if sorted_markers[remaining[0]]['center'][0] > sorted_markers[remaining[1]]['center'][0]:
                top_right_idx, bottom_left_idx = remaining[0], remaining[1]
            else:
                top_right_idx, bottom_left_idx = remaining[1], remaining[0]
            
            diagonal1 = (
                tuple(sorted_markers[top_left_idx]['center']),
                tuple(sorted_markers[bottom_right_idx]['center'])
            )
            diagonal2 = (
                tuple(sorted_markers[top_right_idx]['center']), 
                tuple(sorted_markers[bottom_left_idx]['center'])
            )
            
            intersection_point = line_intersection(diagonal1, diagonal2)
            
            if intersection_point:
                central_point = intersection_point
                
                # Add visualization if debug mode is on
                if debug_display and display_frame is not None:
                    cv2.line(display_frame, diagonal1[0], diagonal1[1], (255, 165, 0), 2)
                    cv2.line(display_frame, diagonal2[0], diagonal2[1], (255, 165, 0), 2)
                    cv2.circle(display_frame, central_point, 8, (0, 0, 255), -1)
            else:
                marker_centers = np.array([marker['center'] for marker in detected_markers])
                central_point = np.mean(marker_centers, axis=0).astype(int)
        else:
            marker_centers = np.array([marker['center'] for marker in detected_markers])
            central_point = np.mean(marker_centers, axis=0).astype(int)
            
        # Add visualization if debug mode is on
        # if debug_display and display_frame is not None:
        #     cv2.drawFrameAxes(display_frame, intrinsic_camera, distortion, 
        #                      central_rvec, central_tvec, 0.1)
        #     cv2.circle(display_frame, central_point, 5, (255, 0, 255), -1)
        #     cv2.putText(display_frame, "CENTRAL POSE", 
        #                (central_point[0]-50, central_point[1]-30),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        #     cv2.putText(display_frame, f"Dist: {central_distance:.3f}m", 
        #                (central_point[0]-50, central_point[1]+20),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

            # Connect markers with lines (except for 4 markers where we draw diagonals)
            if len(detected_markers) != 4:
                for i in range(len(detected_markers)):
                    for j in range(i+1, len(detected_markers)):
                        pt1 = tuple(detected_markers[i]['center'])
                        pt2 = tuple(detected_markers[j]['center'])
                        cv2.line(display_frame, pt1, pt2, (0, 165, 255), 2)
    else:
        # Only one marker detected, use its position
        central_rvec = detected_markers[0]['rvec']
        central_tvec = detected_markers[0]['tvec']
        central_point = detected_markers[0]['center']
        central_distance = detected_markers[0]['distance']
        
        # Add visualization if debug mode is on
        # if debug_display and display_frame is not None:
        #     cv2.drawFrameAxes(display_frame, intrinsic_camera, distortion, 
        #                      central_rvec, central_tvec, 0.1)
        #     cv2.putText(display_frame, f"Single Marker Dist: {central_distance:.3f}m", 
        #               (central_point[0]-50, central_point[1]+20),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Display debug window if requested
    # if debug_display and display_frame is not None:
    #     cv2.imshow("ArUco Detection Debug - Press any key to continue", display_frame)
    #     cv2.waitKey(1)
        # cv2.destroyAllWindows()
    # Return results
    return {
        "success": True,
        "central_rvec": central_rvec,
        "central_tvec": central_tvec,
        "central_point": central_point,
        "central_distance": central_distance,
        "markers": detected_markers,
    }