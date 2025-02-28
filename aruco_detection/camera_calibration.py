import cv2
import numpy as np
import os
import time
import json

def save_calibration(mtx, dist, path="calibration_params.json"):
    """
    Save calibration parameters to a JSON file
    """
    data = {
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.tolist()
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Calibration parameters saved to {path}")

def load_calibration(path="calibration_params.json"):
    """
    Load calibration parameters from a JSON file
    """
    if not os.path.exists(path):
        print(f"Calibration file {path} does not exist")
        return None, None
        
    with open(path, 'r') as f:
        data = json.load(f)
    
    mtx = np.array(data["camera_matrix"])
    dist = np.array(data["distortion_coefficients"])
    
    return mtx, dist

def calibrate_camera(checkerboard_size=(9, 6), square_size=0.025, num_images=20):
    """
    Calibrates the camera using a checkerboard pattern
    
    Parameters:
    - checkerboard_size: The number of internal corners (width, height)
    - square_size: The size of each square in meters
    - num_images: Number of images to capture for calibration
    
    Returns:
    - ret: The RMS error of the calibration
    - mtx: The camera matrix
    - dist: The distortion coefficients
    - rvecs: The rotation vectors
    - tvecs: The translation vectors
    """
    # Define termination criteria for corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points: (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Access webcam
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None, None, None, None, None
    
    # Capture calibration images
    images_captured = 0
    last_capture_time = time.time()
    delay_between_captures = 1  # seconds
    
    print("Camera calibration started. Please hold up the checkerboard pattern.")
    print(f"Capturing {num_images} images for calibration...")
    print("Press 's' to manually capture an image or 'q' to quit.")
    
    while images_captured < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
            
        # Create a copy for drawing
        display = frame.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        # If corners found, add object and image points
        if ret_corners:
            # Draw the corners
            cv2.drawChessboardCorners(display, checkerboard_size, corners, ret_corners)
            
            # Show status
            cv2.putText(display, f"Detected! Images: {images_captured}/{num_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            current_time = time.time()
            if current_time - last_capture_time >= delay_between_captures:
                auto_capture = True
            else:
                auto_capture = False
                
            # Show timer for next auto-capture
            time_left = max(0, delay_between_captures - (current_time - last_capture_time))
            cv2.putText(display, f"Next auto-capture in: {time_left:.1f}s", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            cv2.putText(display, "No checkerboard detected", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, f"Images: {images_captured}/{num_images}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            auto_capture = False
        
        # Display the image
        cv2.imshow("Camera Calibration", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Check for manual capture with 's' key
        if key == ord('s') and ret_corners:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            images_captured += 1
            last_capture_time = time.time()
            print(f"Manually captured image {images_captured}/{num_images}")
        
        # Auto-capture if checkerboard detected and delay passed
        elif ret_corners and auto_capture:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            images_captured += 1
            last_capture_time = time.time()
            print(f"Auto-captured image {images_captured}/{num_images}")
        
        # Exit on 'q'
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # If we captured enough images, calculate camera calibration
    if images_captured >= 5:  # Need at least 5 images for a decent calibration
        print("Calculating calibration parameters...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        print("Calibration complete!")
        print(f"RMS reprojection error: {ret}")
        print("Camera matrix:")
        print(mtx)
        print("Distortion coefficients:")
        print(dist)
        
        # Return calibration results
        return ret, mtx, dist, rvecs, tvecs
    else:
        print("Not enough images captured for calibration.")
        return None, None, None, None, None

def test_calibration(mtx, dist):
    """
    Test the calibration by showing the undistorted camera feed
    """
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Testing calibration. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort the frame using the calibration parameters
        undistorted = cv2.undistort(frame, mtx, dist, None, mtx)
        
        # Display the original and undistorted frames side by side
        comparison = np.hstack((frame, undistorted))
        
        # Resize for display if too large
        h, w = comparison.shape[:2]
        if w > 1600:
            scale = 1600 / w
            comparison = cv2.resize(comparison, (int(w * scale), int(h * scale)))
        
        # Add labels
        half_width = comparison.shape[1] // 2
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(comparison, "Undistorted", (half_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Calibration Test", comparison)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Check if calibration file exists
    calibration_file = "calibration_params.json"
    
    if os.path.exists(calibration_file):
        print(f"Found existing calibration file: {calibration_file}")
        choice = input("Do you want to use the existing calibration (y) or recalibrate (n)? ").lower()
        
        if choice == 'y':
            mtx, dist = load_calibration(calibration_file)
            if mtx is not None and dist is not None:
                print("Loaded existing calibration parameters:")
                print("Camera matrix:")
                print(mtx)
                print("Distortion coefficients:")
                print(dist)
                
                test_choice = input("Do you want to test the calibration (y/n)? ").lower()
                if test_choice == 'y':
                    test_calibration(mtx, dist)
                return
    
    # If we get here, either no calibration file exists or user wants to recalibrate
    print("\nStarting camera calibration...")
    print("Please use a printed checkerboard pattern.")
    print("Default is 9x6 internal corners (10x7 squares).")
    
    # Allow user to set checkerboard size
    try:
        width = int(input("Enter checkerboard width (internal corners, default 9): ") or 9)
        height = int(input("Enter checkerboard height (internal corners, default 6): ") or 6)
        square_size = float(input("Enter square size in meters (default 0.025): ") or 0.025)
        num_images = int(input("Enter number of images to capture (default 20): ") or 20)
    except ValueError:
        print("Invalid input. Using default values.")
        width, height = 9, 6
        square_size = 0.025
        num_images = 20
    
    # Run calibration
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        checkerboard_size=(width, height),
        square_size=square_size,
        num_images=num_images
    )
    
    # Save calibration if successful
    if mtx is not None and dist is not None:
        save_calibration(mtx, dist, calibration_file)
        
        # Test the calibration
        test_choice = input("Do you want to test the calibration (y/n)? ").lower()
        if test_choice == 'y':
            test_calibration(mtx, dist)

if __name__ == "__main__":
    main()
