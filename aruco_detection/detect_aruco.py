import cv2
import numpy as np
import glfw
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pywavefront

from generate_aruco import ARUCO_DICT

width, height = 1280, 720
# Create a GLFW window for OpenGL
if not glfw.init():
    raise Exception("GLFW initialization failed!")

window = glfw.create_window(1280, 720, "Hidden OpenGL Window", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window creation failed!")

glfw.make_context_current(window)

# Enable OpenGL settings
gl.glEnable(gl.GL_DEPTH_TEST)

# Load the 3D model
model = pywavefront.Wavefront("E:\\artest\\AR\\aruco_detection\\cube\\tinker.obj", collect_faces=True)


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

def draw_model():
    gl.glBegin(gl.GL_TRIANGLES)
    for mesh in model.mesh_list:
        for face in mesh.faces:
            for vertex_index in face:
                gl.glVertex3fv(model.vertices[vertex_index])
    gl.glEnd()


def draw_cube_on_marker(rvec, tvec):
    gl.glPushMatrix()

    # Convert rotation vector to rotation matrix.
    rotM, _ = cv2.Rodrigues(rvec)

    print(rotM)

    angle = np.deg2rad(180)  # Experiment with this angle
    R_fix = cv2.Rodrigues(np.array([angle, 0, 0]))[0]
    R_corrected = R_fix @ rotM

    transform_matrix = np.eye(4, dtype=np.float32)
    transform_matrix[:3, :3] = R_corrected
    transform_matrix[:3, 3]  = tvec.squeeze()

    # Fix the coordinate mismatch by flipping Z.
    # (Adjust this fix as needed based on your coordinate conventions.)
    fix = np.diag([-1, -1, -1, 1]).astype(np.float32)
    transform_matrix = fix @ transform_matrix

    # Apply the transformation.
    gl.glMultMatrixf(transform_matrix.T)

    gl.glTranslatef(-0.5, -0.5, -0.5)
    gl.glScalef(0.1, 0.1, 0.1)

    # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    gl.glColor3f(0, 1, 0)
    draw_model()

    # Draw wireframe overlay
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
    gl.glLineWidth(2.0)
    gl.glColor3f(0, 0, 0)
    draw_model()

    # Reset polygon mode
    gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
    
    gl.glPopMatrix()


def create_fbo(width, height):
    fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    
    texture = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                              gl.GL_TEXTURE_2D, texture, 0)
    
    depth_buffer = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
    gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                 gl.GL_RENDERBUFFER, depth_buffer)
    
    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        print("Framebuffer not complete:", status)
    
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return fbo, texture, depth_buffer

def render_ar_overlay(rvec, tvec, width, height, display_frame):
    """
    Renders the AR object using OpenGL and blends it with the camera feed.
    """
    fbo, texture, depth_buffer = create_fbo(width, height)

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glViewport(0, 0, width, height)
    
    # Transparent background
    gl.glClearColor(0, 0, 0, 0)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

    # Setup Perspective Projection
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluPerspective(45, float(width) / height, 0.1, 100.0)

    # Set ModelView with marker pose
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    glu.gluLookAt(0, 0, 3.0, 0, 0, 0, 0, 1.0, 0)

    scale_factor = np.array([-19.0, 18.0, 10.0]).reshape(3, 1)
    tvec *= scale_factor
    draw_cube_on_marker(rvec, tvec)

    # Capture OpenGL output
    gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
    rgba_raw = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    fbo_frame = np.frombuffer(rgba_raw, dtype=np.uint8).reshape(height, width, 4)
    fbo_frame = cv2.flip(fbo_frame, 0)

    # Convert to float [0,1]
    fbo_float = fbo_frame.astype(np.float32) / 255.0
    display_float = display_frame.astype(np.float32) / 255.0

    # Alpha blending
    alpha = fbo_float[..., 3:4]
    fbo_color = fbo_float[..., :3]

    blended_float = fbo_color * alpha + display_float * (1 - alpha)
    blended = (blended_float * 255).astype(np.uint8)

    return blended

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
            1008.6240593223057,
            0.0,
            621.5174070772955
        ],
        [
            0.0,
            1003.0659994333062,
            338.5153361642863
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])
    
    if distortion is None:
        distortion = np.array([
            0.0897805377291951,
            0.06092364616375435,
            -0.0015493723815887572,
            -0.008138973787935298,
            -0.8980223494831121
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
    
    # For visualization if debug_display is True
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
    
    ar_frame = render_ar_overlay(central_rvec, central_tvec, width, height, display_frame)
    # Return results
    return {
        "success": True,
        "central_rvec": central_rvec,
        "central_tvec": central_tvec,
        "central_point": central_point,
        "central_distance": central_distance,
        "markers": detected_markers,
        "frame": ar_frame if debug_display else None
        
    }

def run_ar():
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

        # Call detect_aruco_markers() to process the frame and get AR overlay
        result = detect_aruco_markers(frame=frame, debug_display=True)

        if result["success"]:
            ar_frame = result["frame"]
        else:
            # If no marker is detected, just show the normal camera feed
            ar_frame = frame

        cv2.imshow("AR Camera", ar_frame)

        glfw.poll_events()
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    
if __name__ == "__main__":
    run_ar()