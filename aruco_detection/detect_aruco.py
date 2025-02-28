import cv2
import numpy as np
from generate_aruco import ARUCO_DICT

# def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    
#     parameters = cv2.aruco.DetectorParameters()

#     detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

#     corners, ids, rejected_img_points = detector.detectMarkers(gray)

#     if ids is not None:
#         for i in range(len(ids)):
#             marker_length = 0.1  # Marker size in meters (Change to acutal values)
#             object_points = np.array([
#                 [-marker_length / 2, marker_length / 2, 0],
#                 [marker_length / 2, marker_length / 2, 0],
#                 [marker_length / 2, -marker_length / 2, 0],
#                 [-marker_length / 2, -marker_length / 2, 0]
#             ], dtype=np.float32)

#             success, rvec, tvec = cv2.solvePnP(
#                 object_points, corners[i][0], matrix_coefficients, distortion_coefficients
#             )

#             if success:

#                 cv2.aruco.drawDetectedMarkers(frame, corners)
#                 cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.1)

#     return frame

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def decode_aruco_marker(marker_binary, dict_type="DICT_7X7_1000"):
    """
    Decodes an ArUco marker with error correction
    """
    # Get marker dimensions based on dictionary type
    if "7X7" in dict_type:
        grid_size = 7
    elif "6X6" in dict_type:
        grid_size = 6
    elif "5X5" in dict_type:
        grid_size = 5
    elif "4X4" in dict_type:
        grid_size = 4
    else:
        grid_size = 7  # Default to 7x7
    
    # Calculate cell size in pixels
    cell_size = marker_binary.shape[0] // grid_size
    
    # Initialize grid to store binary values
    grid = np.zeros((grid_size-2, grid_size-2), dtype=np.uint8)
    
    # Extract the inner grid (excluding the border)
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            # Calculate center of current cell
            cell_center_x = j * cell_size + cell_size // 2
            cell_center_y = i * cell_size + cell_size // 2
            
            # Sample a small region around the center (more robust)
            sample_size = cell_size // 3
            cell_region = marker_binary[
                cell_center_y-sample_size:cell_center_y+sample_size,
                cell_center_x-sample_size:cell_center_x+sample_size
            ]
            
            # Determine if cell is black or white
            if cell_region.mean() > 127:  # Adjust threshold if needed
                grid[i-1, j-1] = 1
            else:
                grid[i-1, j-1] = 0
    
    # Implement basic error detection using parity check
    # Assuming the last row and column contain parity bits
    valid = True
    corrections_made = 0
    
    # Check row parity
    for i in range(grid.shape[0]-1):
        row_sum = np.sum(grid[i, :-1]) % 2
        if row_sum != grid[i, -1]:
            valid = False
            corrections_made += 1
            # Attempt correction if only one error found
            if corrections_made <= 1:
                # Find the most likely error position
                for j in range(grid.shape[1]-1):
                    test_grid = grid.copy()
                    test_grid[i, j] = 1 - test_grid[i, j]  # Flip the bit
                    if np.sum(test_grid[i, :-1]) % 2 == grid[i, -1]:
                        grid[i, j] = test_grid[i, j]  # Apply correction
                        break
    
    # Check column parity
    for j in range(grid.shape[1]-1):
        col_sum = np.sum(grid[:-1, j]) % 2
        if col_sum != grid[-1, j]:
            valid = False
            corrections_made += 1
            # Attempt correction if only one error found
            if corrections_made <= 1:
                # Find the most likely error position
                for i in range(grid.shape[0]-1):
                    test_grid = grid.copy()
                    test_grid[i, j] = 1 - test_grid[i, j]  # Flip the bit
                    if np.sum(test_grid[:-1, j]) % 2 == grid[-1, j]:
                        grid[i, j] = test_grid[i, j]  # Apply correction
                        break
    
    # Extract marker ID from the inner bits (excluding parity bits)
    marker_bits = grid[:-1, :-1].flatten()
    marker_id = 0
    for bit in marker_bits:
        marker_id = (marker_id << 1) | bit
    
    return marker_id, valid or corrections_made <= 1  # True if valid or corrected

def detect_aruco_manual():
    marker_type = "DICT_7X7_1000" # better on far marker (more precision)
    aruco_dict_type = ARUCO_DICT[marker_type]
    
    # mac cam parameters
    # intrinsic_camera = np.array([
    #     [953.78782288, 0.0, 642.36740727],
    #     [0.0, 953.48758172, 349.33238922],
    #     [0.0, 0.0, 1.0]
    #     ])
    # distortion = np.array([ 4.76145886e-02, -2.41452809e-02, 9.83416329e-05, -2.73093975e-03, 1.72122111e-02])

    #asus cam parameters
    intrinsic_camera = np.array([
        [938.51762171, 0, 617.83801429],
        [0, 940.68786111, 316.6695038 ],
        [0, 0, 1]
    ])
    distortion = np.array([0.02394387, -0.11535456, -0.01334333, -0.00231512,  0.22218833])
    
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

                # prevent zero division that from width_marker / height_marker
                if width_marker > 1e-6 and height_marker > 1e-6: # using small epsilon instead of exact zero
                    aspect_ratio = width_marker / height_marker
                    
                    if 0.8 <= aspect_ratio <= 1.2:
                        dst_size = 100
                        dst_points = np.array([
                            [0, 0],
                            [dst_size-1, 0],
                            [dst_size-1, dst_size-1],
                            [0, dst_size-1]
                        ], dtype=np.float32)
                        
                        # get perspective of auco marker
                        M = cv2.getPerspectiveTransform(rect, dst_points)
                        warped = cv2.warpPerspective(gray, M, (dst_size, dst_size))
                        
                        # threshold the warped image
                        _, marker_binary = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                        border = 10
                        border_pixels = np.sum(marker_binary[:border, :]) + np.sum(marker_binary[-border:, :]) + \
                                      np.sum(marker_binary[:, :border]) + np.sum(marker_binary[:, -border:])
                        border_avg = border_pixels / (4 * border * dst_size)
                        
                        if border_avg > 200:
                            # Try to decode the marker with error correction
                            marker_id, is_valid = decode_aruco_marker(marker_binary, marker_type)
                            
                            if is_valid:
                                marker_length = 0.100 # real marker size
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
                                        # draw the detected marker
                                        cv2.polylines(display_frame, [np.int32(rect)], True, (0, 255, 0), 2)
                                        
                                        # draw coordinate axes
                                        cv2.drawFrameAxes(display_frame, intrinsic_camera, 
                                                         distortion, rvec, tvec, 0.05)
                                        
                                        # add marker type and ID on opencv window
                                        center = np.mean(rect, axis=0).astype(int)
                                        cv2.putText(display_frame, f"ArUco ID: {marker_id}", 
                                                   (center[0]-40, center[1]-30),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                        
                                        distance = np.linalg.norm(tvec) * 0.75
                                        cv2.putText(display_frame, f"Dist: {distance:.3f}m",
                                                   (center[0]-30, center[1]-10),
                                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                        
                                except cv2.error:
                                    pass
                else:
                    continue

        cv2.imshow("Manual ArUco Detection - Press q to exit", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_aruco_manual()