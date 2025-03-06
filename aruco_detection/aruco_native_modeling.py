import cv2
import numpy as np
import glfw
import OpenGL.GL as gl
import OpenGL.GLU as glu
import pywavefront
from stl import mesh  # Add this import for STL support
import time
import os

# Configuration
MODEL_PATH = "E:\\artest\\AR\\aruco_detection\\cube\\Mickey Mouse.obj"
WIDTH, HEIGHT = 1280, 720
MARKER_LENGTH = 0.0485  # ArUco marker size in meters
ARUCO_DICT = cv2.aruco.DICT_7X7_1000  # Same dictionary as in the manual detection

# Camera parameters - update with your calibrated values for best results
# Mac parameters
# CAMERA_MATRIX = np.array([
#     [957.1108018720613, 0.0, 556.0882651177826],
#     [0.0, 951.9753671508217, 286.42509589693657],
#     [0.0, 0.0, 1.0]
# ])

# DISTORTION_COEFFS = np.array([
#     -0.25856927733393603, 1.8456432127404514, -0.021219826734632862,
#     -0.024902070756342175, -3.808238876719984
# ])

# Nano Parameters
CAMERA_MATRIX = np.array([
        [
            967.229637877688,
            0.0,
            650.0636729430692
        ],
        [
            0.0,
            978.2884296325454,
            279.4385529511379
        ],
        [
            0.0,
            0.0,
            1.0
        ]
    ])

DISTORTION_COEFFS = np.array([
            0.061815425198978924,
            -0.5653239478518806,
            -0.033745222064303845,
            0.006614884900641152,
            1.4894642088388304
        ])

# Model unit configuration
MODEL_UNITS = 'millimeters'
UNIT_CONVERSION = {
    'millimeters': 0.001,
    'centimeters': 0.01,
    'inches': 0.0254,
    'meters': 1.0
}

def load_stl_as_vertices_and_faces(stl_path):
    """Convert STL file to vertices and faces format compatible with our renderer"""
    stl_mesh = mesh.Mesh.from_file(stl_path)
    
    # Extract unique vertices and build faces
    vertices = stl_mesh.vectors.reshape(-1, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)
    
    # Create a simple class to match pywavefront's interface
    class STLMesh:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.mesh_list = [type('Mesh', (), {'faces': faces})]
    
    return STLMesh(vertices, faces)

class NativeArucoRenderer:
    def __init__(self, model_path=MODEL_PATH, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height
        self.model_path = model_path
        self.camera_matrix = CAMERA_MATRIX
        self.dist_coeffs = DISTORTION_COEFFS
        self.marker_length = MARKER_LENGTH
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Performance tracking
        self.last_detection_time = 0
        self.last_valid_rvecs = None
        self.last_valid_tvecs = None
        self.fps_history = []
        self.detection_history = []  # Track detection stability
        
        # Unit conversion factor
        self.unit_scale = UNIT_CONVERSION.get(MODEL_UNITS, 1.0)
        
        # Add display list ID for model caching
        self.model_display_list = None
        
        # Initialize OpenGL and load 3D model
        self.initialize_gl()
        try:
            # Load model based on file extension
            file_ext = os.path.splitext(model_path)[1].lower()
            if file_ext == '.stl':
                self.model = load_stl_as_vertices_and_faces(model_path)
            else:  # .obj files
                self.model = pywavefront.Wavefront(model_path, collect_faces=True)
            
            print(f"Loaded 3D model: {model_path}")
            self.analyze_model()
            self.create_model_display_list()  # Create display list after loading model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def analyze_model(self):
        """Analyze model dimensions and compute necessary transformations"""
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
        
        # Print model information
        print(f"Model dimensions (original units): {self.dimensions}")
        print(f"Model center: {self.model_center}")
        print(f"Model diagonal size: {self.diagonal_size} {MODEL_UNITS}")
        print(f"Real dimensions (meters): {self.real_dimensions}")
        print(f"Real diagonal size: {self.real_diagonal} meters")
        print(f"Unit scale factor: {self.unit_scale} (from {MODEL_UNITS} to meters)")
        print(f"Reference marker size: {self.marker_length} meters")
        
        # Calculate an offset to place the bottom of the model on the marker plane
        # Assuming Y is up in the model coordinate system
        self.y_offset = self.min_bounds[1]
            
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

    def create_model_display_list(self):
        """Create an OpenGL display list for the model"""
        # Generate a new display list
        self.model_display_list = gl.glGenLists(1)
        gl.glNewList(self.model_display_list, gl.GL_COMPILE)
        
        # Draw the model
        gl.glBegin(gl.GL_TRIANGLES)
        for mesh in self.model.mesh_list:
            for face in mesh.faces:
                for vertex_index in face:
                    gl.glVertex3fv(self.model.vertices[vertex_index])
        gl.glEnd()
        
        gl.glEndList()

    def draw_model(self):
        """Draw the cached model using display list"""
        if self.model_display_list:
            gl.glCallList(self.model_display_list)

    def draw_on_marker(self, rvec, tvec):
        """Draw the 3D model at the position and orientation determined by rvec and tvec"""
        gl.glPushMatrix()

        # Convert rotation vector to rotation matrix
        rotM, _ = cv2.Rodrigues(rvec)

        # Create transformation matrix
        transform_matrix = np.eye(4, dtype=np.float32)
        transform_matrix[:3, :3] = rotM
        transform_matrix[:3, 3] = tvec.flatten()

        # Fix coordinate system mismatch between OpenCV and OpenGL
        coordinate_fix = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],  # Flip Y axis
            [0,  0, -1, 0],  # Flip Z axis
            [0,  0,  0, 1]
        ])
        transform_matrix = coordinate_fix @ transform_matrix

        # Apply the transformation matrix
        gl.glMultMatrixf(transform_matrix.T)
        
        # Scale the model to convert from model units to meters
        MODEL_SCALE = 1.1
        gl.glScalef(self.unit_scale * MODEL_SCALE, self.unit_scale * MODEL_SCALE, self.unit_scale * MODEL_SCALE)
        
        # Draw filled model
        gl.glColor4f(0, 1, 0, 0.9)  # More opaque green
        self.draw_model()

        # Draw wireframe overlay
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(2.0)
        gl.glColor4f(0, 0, 0, 1.0)
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
        
        # Calculate FOV from camera matrix for better perspective matching
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        fov_y = 2 * np.arctan(self.height / (2 * fy)) * 180.0 / np.pi
        
        aspect = self.width / self.height
        # Use a smaller near plane to allow seeing close objects
        glu.gluPerspective(fov_y, aspect, 0.01, 100.0)

        # Set model-view matrix for camera position
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # Draw the model with real-world scaling
        self.draw_on_marker(rvec, tvec)

        # Capture OpenGL output
        gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
        rgba_raw = gl.glReadPixels(0, 0, self.width, self.height, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Delete FBO resources
        try:
            # Try both deletion methods to handle different PyOpenGL versions
            try:
                gl.glDeleteFramebuffers(fbo)
                gl.glDeleteTextures(texture)
                gl.glDeleteRenderbuffers(depth_buffer)
            except:
                gl.glDeleteFramebuffers(1, [fbo])
                gl.glDeleteTextures(1, [texture])
                gl.glDeleteRenderbuffers(1, [depth_buffer])
        except Exception as e:
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
        
    def calculate_central_pose(self, rvecs, tvecs):
        """Calculate a central pose from multiple detected markers"""
        if len(rvecs) == 1:
            return rvecs[0], tvecs[0]
            
        # Average translation vectors
        central_tvec = np.mean(tvecs, axis=0)
        
        # Average rotation vectors (this is a simplification - proper averaging would use quaternions)
        central_rvec = np.mean(rvecs, axis=0)
        
        return central_rvec, central_tvec

    def detect_and_render(self, frame):
        """Detect ArUco markers with native library and render 3D model"""
        start_time = time.time()
        display_frame = frame.copy()
        
        # Convert to grayscale for marker detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        # If markers are detected
        if ids is not None and len(ids) > 0:
            self.detection_history.append(1)  # Detected
            
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
            
            # Estimate pose for each marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs
            )
            
            # Store the latest valid detection
            self.last_detection_time = time.time()
            self.last_valid_rvecs = rvecs
            self.last_valid_tvecs = tvecs
            
            # Calculate distances for each marker
            distances = [np.linalg.norm(tvec) for tvec in tvecs]
            avg_distance = np.mean(distances)
            
            # Draw axes for each detected marker
            for i in range(len(ids)):
                cv2.drawFrameAxes(display_frame, self.camera_matrix, 
                                 self.dist_coeffs, rvecs[i], tvecs[i], 0.05)
                
                # Draw marker ID and distance
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))
                cv2.putText(display_frame, f"ID: {ids[i][0]}, D: {distances[i]:.2f}m", 
                           (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate central pose from all detected markers
            central_rvec, central_tvec = self.calculate_central_pose(rvecs, tvecs)
            
            # Render the 3D model on the central pose
            ar_frame = self.render_ar_overlay(central_rvec, central_tvec, display_frame)
            
            # Display information
            cv2.putText(ar_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(ar_frame, f"Markers: {len(ids)}, Avg Dist: {avg_distance:.3f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(ar_frame, f"Model: {self.real_dimensions[0]:.2f}x{self.real_dimensions[1]:.2f}x{self.real_dimensions[2]:.2f}m", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate detection stability
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            stability = sum(self.detection_history) / len(self.detection_history) * 100
            cv2.putText(ar_frame, f"Stability: {stability:.1f}%", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return ar_frame
            
        else:
            self.detection_history.append(0)  # Not detected
            
            # If we have a recent detection, use that
            if self.last_valid_rvecs is not None and time.time() - self.last_detection_time < 0.5:
                # Use the last valid central pose
                central_rvec, central_tvec = self.calculate_central_pose(
                    self.last_valid_rvecs, self.last_valid_tvecs
                )
                
                # Render using the last known pose
                ar_frame = self.render_ar_overlay(central_rvec, central_tvec, display_frame)
                
                # Show "using last detection" message
                cv2.putText(ar_frame, "Using last detection", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                return ar_frame
                
            # Show FPS even when no markers are detected
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, "No markers detected", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            # Calculate detection stability
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            stability = sum(self.detection_history) / len(self.detection_history) * 100
            cv2.putText(display_frame, f"Stability: {stability:.1f}%", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
            return display_frame

    def run(self):
        """Main AR visualization loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        print("Running Native ArUco AR visualization. Press 'q' to quit.")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    break

                # Process the frame with ArUco detection and 3D rendering
                ar_frame = self.detect_and_render(frame)

                # Display the result
                cv2.imshow("Native ArUco AR", ar_frame)

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
        renderer = NativeArucoRenderer()
        renderer.run()
    except Exception as e:
        print(f"Error: {e}")
        glfw.terminate()
