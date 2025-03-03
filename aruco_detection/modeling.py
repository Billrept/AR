import glfw
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pywavefront
import numpy as np
import cv2
import time
import os
from detect_aruco import detect_aruco_markers

# Configuration
MODEL_PATH = "/Users/phacharakimpha/comp_vision/ass/aruco_detection/cube/tinker.obj"
WIDTH, HEIGHT = 1280, 720
MARKER_LENGTH = 0.049  # Make sure this matches exactly what's used in detect_aruco.py
USE_CPU_RENDERING = False  # Set to True to use CPU-based rendering instead of OpenGL

MODEL_UNITS = 'millimeters'
UNIT_CONVERSION = {
    'millimeters': 0.001,
    'centimeters': 0.01,
    'inches': 0.0254,
    'meters': 1.0
}

class ARModelRenderer:
    def __init__(self, model_path=MODEL_PATH, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.model_path = model_path
        self.last_detection_time = 0
        self.last_valid_rvec = None
        self.last_valid_tvec = None
        self.use_cpu_rendering = USE_CPU_RENDERING
        
        # Unit conversion factor - from model units to meters
        self.unit_scale = UNIT_CONVERSION.get(MODEL_UNITS, 1.0)
        
        # Store camera matrix that matches detect_aruco.py
        self.camera_matrix = np.array([
            [957.1108018720613, 0.0, 556.0882651177826],
            [0.0, 951.9753671508217, 286.42509589693657],
            [0.0, 0.0, 1.0]
        ])
        
        self.distortion = np.array([
            -0.25856927733393603, 1.8456432127404514, -0.021219826734632862,
            -0.024902070756342175, -3.808238876719984
        ])
        
        # Initialize model data
        if self.use_cpu_rendering:
            # CPU-based rendering uses our custom OBJ loader
            self.obj_data = self.load_obj(model_path)
            self.analyze_obj_data()
        else:
            # OpenGL rendering uses pywavefront
            self.initialize_gl()
            try:
                self.model = pywavefront.Wavefront(model_path, collect_faces=True)
                print(f"Loaded 3D model: {model_path}")
                self.analyze_model()
            except Exception as e:
                print(f"Error loading model: {e}")
                raise

    def load_obj(self, filename):
        """Load an OBJ file directly (for CPU-based rendering)"""
        print(f"Loading model from: {filename}")
        vertices = []
        faces = []
        
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file not found: {filename}")
        
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
                    
        obj_data = {
            'vertices': np.array(vertices, dtype=np.float32),
            'faces': faces
        }
        print(f"Loaded model with {len(vertices)} vertices and {len(faces)} faces")
        return obj_data

    def analyze_obj_data(self):
        """Analyze the model dimensions for the CPU rendering path"""
        vertices = self.obj_data['vertices']
        
        # Calculate bounds and dimensions
        self.min_bounds = np.min(vertices, axis=0)
        self.max_bounds = np.max(vertices, axis=0)
        self.dimensions = self.max_bounds - self.min_bounds
        
        # Calculate model center and size
        self.model_center = (self.min_bounds + self.max_bounds) / 2
        self.diagonal_size = np.linalg.norm(self.dimensions)
        
        # Calculate real-world dimensions (in meters)
        self.real_dimensions = self.dimensions * self.unit_scale
        self.real_diagonal = self.diagonal_size * self.unit_scale
        
        # Apply scaling to vertices (in place)
        self.obj_data['vertices'] *= self.unit_scale
        
        # Print model information
        print(f"Model dimensions (original units): {self.dimensions}")
        print(f"Model center: {self.model_center}")
        print(f"Real dimensions (meters): {self.real_dimensions}")
        
        # Calculate an offset to place the bottom of the model on the marker plane
        # Typically Y is up in the model coordinate system
        self.y_offset = self.min_bounds[1]

    def analyze_model(self):
        """Analyze model dimensions and compute necessary transformations for accurate display"""
        vertices = np.array(self.model.vertices)
        
        # Calculate bounds and dimensions
        self.min_bounds = np.min(vertices, axis=0)
        self.max_bounds = np.max(vertices, axis=0)
        self.dimensions = self.max_bounds - self.min_bounds
        
        # Calculate model center and size
        self.model_center = (self.min_bounds + self.max_bounds) / 2
        self.diagonal_size = np.linalg.norm(self.dimensions)
        
        # Calculate real-world dimensions (in meters)
        self.real_dimensions = self.dimensions * self.unit_scale
        self.real_diagonal = self.diagonal_size * self.unit_scale
        
        # Determine the bottom of the model
        # Y-up is common in 3D models, but verify with your model
        if self.dimensions[1] > self.dimensions[2]:
            # Y is likely the up axis
            self.up_axis = 1
            self.bottom_offset = self.min_bounds[1]
        else:
            # Z is likely the up axis
            self.up_axis = 2
            self.bottom_offset = self.min_bounds[2]
        
        print(f"Model dimensions (original units): {self.dimensions}")
        print(f"Model center: {self.model_center}")
        print(f"Model diagonal size: {self.diagonal_size} {MODEL_UNITS}")
        print(f"Real dimensions (meters): {self.real_dimensions}")
        print(f"Real diagonal size: {self.real_diagonal} meters")
        print(f"Unit scale factor: {self.unit_scale} (from {MODEL_UNITS} to meters)")
        print(f"Reference marker size: {MARKER_LENGTH} meters")
        print(f"Determined up axis: {['X', 'Y', 'Z'][self.up_axis]}")
        print(f"Model bottom offset: {self.bottom_offset}")

    def initialize_gl(self):
        """Initialize OpenGL context and settings"""
        if not glfw.init():
            raise Exception("GLFW initialization failed!")

        # Create a window but make it invisible (we only need the OpenGL context)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(self.width, self.height, "Hidden OpenGL Window", None, None)
        
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window creation failed!")

        glfw.make_context_current(self.window)

        # Enable OpenGL settings
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

    def draw_model(self):
        """Draw the 3D model using its faces"""
        gl.glBegin(gl.GL_TRIANGLES)
        for mesh in self.model.mesh_list:
            for face in mesh.faces:
                for vertex_index in face:
                    gl.glVertex3fv(self.model.vertices[vertex_index])
        gl.glEnd()

    def draw_cube_on_marker(self, rvec, tvec):
        """Draw the 3D model at the position and orientation determined by rvec and tvec"""
        gl.glPushMatrix()

        # Convert rotation vector to rotation matrix
        rotM, _ = cv2.Rodrigues(rvec)
        
        # Debug marker position - useful to identify issues
        print(f"Marker position (tvec): {tvec.flatten()}")
        
        # Important: tvec from ArUco detection is in camera coordinates
        # We need to convert it to OpenGL world coordinates
        
        # First, create the OpenCV coordinate transformation matrix
        transform_matrix = np.eye(4, dtype=np.float32)
        transform_matrix[:3, :3] = rotM
        transform_matrix[:3, 3] = tvec.flatten()
        
        # Convert OpenCV coordinates to OpenGL coordinates
        # OpenCV: +X right, +Y down, +Z forward (looking from camera)
        # OpenGL: +X right, +Y up, +Z backward (away from camera)
        opencv_to_opengl = np.array([
            [1.0,  0.0,  0.0, 0.0],  # X stays the same
            [0.0, -1.0,  0.0, 0.0],  # Y is flipped
            [0.0,  0.0, -1.0, 0.0],  # Z is flipped
            [0.0,  0.0,  0.0, 1.0]
        ])
        
        # Apply the coordinate system conversion
        modelview = opencv_to_opengl @ transform_matrix
        
        # Apply the transformation matrix
        gl.glMultMatrixf(modelview.T)  # OpenGL uses column-major order
        
        # Scale the model to convert from model units to meters
        gl.glScalef(self.unit_scale, self.unit_scale, self.unit_scale)
        
        # Center the model horizontally but place it directly on the marker plane
        gl.glTranslatef(-self.model_center[0], 0, -self.model_center[2])
        
        # Make sure the bottom of the model sits exactly on the marker plane
        if hasattr(self, 'up_axis'):
            if self.up_axis == 1:  # Y is up
                gl.glTranslatef(0, -self.min_bounds[1], 0)
            else:  # Z is up
                gl.glTranslatef(0, 0, -self.min_bounds[2])
        else:
            # Default to Y-up if up_axis not determined
            gl.glTranslatef(0, -self.min_bounds[1], 0)
            
        # Draw filled model with semi-transparency
        gl.glColor4f(0, 1, 0, 0.8)
        self.draw_model()
        
        # Draw wireframe overlay
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(2.0)
        gl.glColor3f(0, 0, 0)
        self.draw_model()
        
        # Reset polygon mode
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
        
        gl.glPopMatrix()

    def create_fbo(self):
        """Create a framebuffer object for offscreen rendering"""
        fbo = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, self.width, self.height, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                gl.GL_TEXTURE_2D, texture, 0)
        
        depth_buffer = gl.glGenRenderbuffers(1)
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_buffer)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, self.width, self.height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
                                    gl.GL_RENDERBUFFER, depth_buffer)
        
        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            print("Framebuffer not complete:", status)
        
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        return fbo, texture, depth_buffer

    def render_ar_overlay(self, rvec, tvec, display_frame):
        """Render the 3D model onto the camera frame using the detected marker pose"""
        fbo, texture, depth_buffer = self.create_fbo()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
        gl.glViewport(0, 0, self.width, self.height)
        
        # Transparent background
        gl.glClearColor(0, 0, 0, 0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Setup perspective projection to match camera
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        
        # Calculate FOV from camera intrinsics for accurate perspective
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        
        # Calculate field of view for OpenGL perspective
        fov_y = 2 * np.arctan(self.height / (2 * fy)) * 180.0 / np.pi
        
        # Set up perspective projection
        aspect = self.width / self.height
        near_plane = 0.01  # Close enough for nearby objects
        far_plane = 100.0  # Far enough for distant objects
        glu.gluPerspective(fov_y, aspect, near_plane, far_plane)
        
        # Setup camera position (identity for model-view)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Draw model with accurate positioning and scaling
        self.draw_cube_on_marker(rvec, tvec)

        # Capture OpenGL output
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        rgba_raw = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Delete FBO resources - using the correct PyOpenGL syntax
        # Different PyOpenGL versions handle resource deletion differently
        try:
            # Method 1: Try with single parameter (the resource ID)
            gl.glDeleteFramebuffers(fbo)
            gl.glDeleteTextures(texture)
            gl.glDeleteRenderbuffers(depth_buffer)
        except TypeError:
            # Method 2: If that fails, try with array notation
            try:
                gl.glDeleteFramebuffers([fbo])
                gl.glDeleteTextures([texture])
                gl.glDeleteRenderbuffers([depth_buffer])
            except Exception as e:
                # If both methods fail, just continue without deleting
                # This will cause a memory leak but the app can still run
                print(f"Warning: Failed to delete OpenGL resources: {e}")

        # Convert the raw pixel data to a numpy array
        fbo_frame = np.frombuffer(rgba_raw, dtype=np.uint8).reshape(self.height, self.width, 4)
        fbo_frame = cv2.flip(fbo_frame, 0)  # Flip vertically (OpenGL has origin at bottom-left)

        # Alpha blending between the OpenGL render and camera frame
        fbo_float = fbo_frame.astype(np.float32) / 255.0
        display_float = display_frame.astype(np.float32) / 255.0

        # Use alpha channel for blending
        alpha = fbo_float[..., 3:4]
        fbo_color = fbo_float[..., :3]

        blended_float = fbo_color * alpha + display_float * (1 - alpha)
        blended = (blended_float * 255).astype(np.uint8)

        return blended

    def project_obj_solid(self, frame, rvec, tvec, camera_matrix, dist_coeffs):
        """Project 3D model onto the frame using CPU-based rendering with depth sorting"""
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        vertices_obj = self.obj_data['vertices']
        faces = self.obj_data['faces']

        # Calculate the positions of vertices in camera space
        vertices_cam = []
        for pt in vertices_obj:
            pt3d = pt.reshape(3, 1)
            pt_cam = R @ pt3d + tvec
            vertices_cam.append(pt_cam.flatten())
        vertices_cam = np.array(vertices_cam)

        # Project 3D points to 2D image coordinates
        projected_pts_2d, _ = cv2.projectPoints(vertices_obj, rvec, tvec, camera_matrix, dist_coeffs)
        projected_pts_2d = projected_pts_2d.reshape(-1, 2).astype(int)

        # Sort faces by depth (furthest first for proper occlusion)
        face_depths = []
        for f_idx, face in enumerate(faces):
            if len(face) >= 3:  # Ensure we have at least a triangle
                # Average Z depth of the face vertices
                cam_z = [vertices_cam[idx][2] for idx in face]
                avg_z = np.mean(cam_z)
                face_depths.append((f_idx, avg_z))
        
        # Sort faces by Z depth, drawing from back to front
        face_depths.sort(key=lambda x: x[1], reverse=True)

        # Create a copy of the frame for rendering
        result_frame = frame.copy()

        # Draw the faces
        for (face_idx, _) in face_depths:
            idxs = faces[face_idx]
            if len(idxs) >= 3:  # Ensure face has at least 3 vertices
                # Extract 2D coordinates for this face
                face_2d = np.array([projected_pts_2d[idx] for idx in idxs])
                face_2d = face_2d.reshape(-1, 1, 2)
                
                # Draw the filled face
                cv2.fillConvexPoly(result_frame, face_2d, (0, 200, 0), lineType=cv2.LINE_AA)
                
                # Draw the wireframe edges
                cv2.polylines(result_frame, [face_2d], True, (0, 0, 0), 1, cv2.LINE_AA)

        return result_frame

    def process_frame(self, frame):
        """Process a camera frame, detect markers, and render the AR overlay"""
        # Make a copy of the input frame for display
        display_frame = frame.copy()
        
        # Call detect_aruco_markers with explicit camera parameters
        result = detect_aruco_markers(
            frame=frame, 
            debug_display=False,
            intrinsic_camera=self.camera_matrix,
            distortion=self.distortion,
            marker_length=MARKER_LENGTH
        )
        
        # If markers were successfully detected
        if result["success"]:
            # Add debugging output to help diagnose positioning issues
            print(f"Central tvec from detection: {result['central_tvec'].flatten()}")
            print(f"Central rvec from detection: {result['central_rvec'].flatten()}")
            print(f"Distance: {result['central_distance']} meters")
            
            self.last_detection_time = time.time()
            self.last_valid_rvec = result["central_rvec"]
            self.last_valid_tvec = result["central_tvec"]
            
            if self.use_cpu_rendering:
                # Use CPU-based rendering
                # Get camera matrix from result or use approximation
                camera_matrix = np.array([
                    [957.1108018720613, 0.0, 556.0882651177826],
                    [0.0, 951.9753671508217, 286.42509589693657],
                    [0.0, 0.0, 1.0]
                ])
                distortion = np.array([
                    -0.25856927733393603, 1.8456432127404514, -0.021219826734632862,
                    -0.024902070756342175, -3.808238876719984
                ])
                
                # Render model using CPU-based method
                ar_frame = self.project_obj_solid(
                    display_frame, 
                    result["central_rvec"], 
                    result["central_tvec"],
                    camera_matrix, 
                    distortion
                )
            else:
                # Use OpenGL-based rendering
                ar_frame = self.render_ar_overlay(
                    result["central_rvec"], 
                    result["central_tvec"], 
                    display_frame
                )
            
            # Add information about the detected markers
            if "central_distance" in result:
                cv2.putText(
                    ar_frame, 
                    f"Distance: {result['central_distance']:.3f}m", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
            
            # Add information about the model's physical dimensions
            cv2.putText(
                ar_frame,
                f"Model size: {self.real_dimensions[0]:.2f}x{self.real_dimensions[1]:.2f}x{self.real_dimensions[2]:.2f}m",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Add rendering mode info
            mode_text = "CPU Rendering" if self.use_cpu_rendering else "GPU Rendering"
            cv2.putText(
                ar_frame,
                mode_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )
            
            return ar_frame
        else:
            # If detection failed but we have a recent valid detection, use that
            if self.last_valid_rvec is not None and time.time() - self.last_detection_time < 1.0:
                # Use the last valid pose with a small decay effect
                ar_frame = self.render_ar_overlay(
                    self.last_valid_rvec,
                    self.last_valid_tvec,
                    display_frame
                )
                
                # Add a status message
                cv2.putText(
                    ar_frame, 
                    "Using last detection", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 255), 
                    2
                )
                
                return ar_frame
            
            # If no recent detection is available, just return the original frame
            cv2.putText(
                display_frame, 
                "No markers detected", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            return display_frame

    def run(self):
        """Main loop to capture camera frames and display the AR overlay"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Running AR visualization. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Process the frame and get the AR overlay
                ar_frame = self.process_frame(frame)

                # Display the result
                cv2.imshow("AR Visualization", ar_frame)

                # Check for key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()
            glfw.terminate()

if __name__ == "__main__":
    try:
        renderer = ARModelRenderer()
        renderer.run()
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()

