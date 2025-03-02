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
    
    # Check if lines are parallel
    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if abs(denom) < 1e-10:  # Parallel lines
        return None
    
    # Calculate intersection point
    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    x = x1 + ua * (x2 - x1)
    y = y1 + ua * (y2 - y1)
    
    if (0 <= ua <= 1):
        return (int(x), int(y))
    else:
        return (int(x), int(y))

def detect_aruco_manual():
    marker_type = "DICT_7X7_1000" # better on far marker (more precision)
    aruco_dict_type = ARUCO_DICT[marker_type]
    
    # mac cam parameters
    intrinsic_camera = np.array([
        [942.9176688, 0.0, 610.320383],
        [0.0, 941.1945546, 378.8276605],
        [0.0, 0.0, 1.0]
        ])
    distortion = np.array([ 0.06040788, 0.25311179, 0.00882055, -0.01687159, -0.94299114])
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        h, w, _ = frame.shape
        width = 1000
        height = int(width * (h / w))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
        display_frame = frame.copy()
        
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
                            marker_length = 0.049 # real marker size in meters
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
                                    
                                    cv2.polylines(display_frame, [np.int32(rect)], True, (0, 255, 0), 2)
                                    cv2.drawFrameAxes(display_frame, intrinsic_camera, 
                                                     distortion, rvec, tvec, 0.05)
                                    
                                    cv2.putText(display_frame, f"ArUco", 
                                               (center[0]-20, center[1]-30),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    cv2.putText(display_frame, f"Dist: {distance:.3f}m",
                                               (center[0]-30, center[1]-10),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    
                            except cv2.error:
                                pass
                else:
                    continue
        
        # Calculate central pose if markers are detected
        if len(detected_markers) >= 2:
            # Calculate average position for tvec and rvec (3D pose)
            tvecs = [marker['tvec'] for marker in detected_markers]
            central_tvec = np.mean(tvecs, axis=0)
            
            rvecs = [marker['rvec'] for marker in detected_markers]
            central_rvec = np.mean(rvecs, axis=0)
            
            # For 4 markers, find center by intersection of diagonals
            if len(detected_markers) == 4:
                # Sort markers by position to identify corners
                sorted_markers = sorted(detected_markers, key=lambda m: m['center'][0])
                
                top_left_idx = np.argmin([m['center'][0] + m['center'][1] for m in sorted_markers])
                bottom_right_idx = np.argmax([m['center'][0] + m['center'][1] for m in sorted_markers])
                
                remaining = [i for i in range(4) if i != top_left_idx and i != bottom_right_idx]
                if sorted_markers[remaining[0]]['center'][0] > sorted_markers[remaining[1]]['center'][0]:
                    top_right_idx, bottom_left_idx = remaining[0], remaining[1]
                else:
                    top_right_idx, bottom_left_idx = remaining[1], remaining[0]
                
                # Create diagonals
                diagonal1 = (
                    tuple(sorted_markers[top_left_idx]['center']),
                    tuple(sorted_markers[bottom_right_idx]['center'])
                )
                diagonal2 = (
                    tuple(sorted_markers[top_right_idx]['center']), 
                    tuple(sorted_markers[bottom_left_idx]['center'])
                )
                
                # Find intersection
                intersection_point = line_intersection(diagonal1, diagonal2)
                
                if intersection_point:
                    central_point = intersection_point
                    
                    # Draw diagonals connecting opposite markers
                    cv2.line(display_frame, diagonal1[0], diagonal1[1], (255, 165, 0), 2)
                    cv2.line(display_frame, diagonal2[0], diagonal2[1], (255, 165, 0), 2)
                    
                    cv2.circle(display_frame, central_point, 8, (0, 0, 255), -1)
                else:
                    # Fallback to average point if no valid intersection
                    marker_centers = np.array([marker['center'] for marker in detected_markers])
                    central_point = np.mean(marker_centers, axis=0).astype(int)
            else:
                # For 2, 3, or 5+ markers, use the average center
                marker_centers = np.array([marker['center'] for marker in detected_markers])
                central_point = np.mean(marker_centers, axis=0).astype(int)
            
            # Draw central pose axes
            cv2.drawFrameAxes(display_frame, intrinsic_camera, distortion, 
                             central_rvec, central_tvec, 0.1)
            
            # Calculate average distance
            central_distance = np.linalg.norm(central_tvec) * 0.75
            
            # Draw central point and information
            cv2.circle(display_frame, central_point, 5, (255, 0, 255), -1)
            cv2.putText(display_frame, "CENTRAL POSE", 
                       (central_point[0]-50, central_point[1]-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(display_frame, f"Dist: {central_distance:.3f}m", 
                       (central_point[0]-50, central_point[1]+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # Connect markers with lines (except for 4 markers where we draw diagonals)
            if len(detected_markers) != 4:
                for i in range(len(detected_markers)):
                    for j in range(i+1, len(detected_markers)):
                        pt1 = tuple(detected_markers[i]['center'])
                        pt2 = tuple(detected_markers[j]['center'])
                        cv2.line(display_frame, pt1, pt2, (0, 165, 255), 2)

        cv2.imshow("Manual ArUco Detection - Press q to exit", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_manual()