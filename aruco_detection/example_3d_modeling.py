import numpy as np
import cv2
import trimesh
from detect_aruco import detect_aruco_markers
import pyrender
import time
from threading import Thread
import platform
import queue

class ArUcoModelVisualizer:
    def __init__(self, model_path='/Users/phacharakimpha/comp_vision/ass/aruco_detection/cube/Terrific Borwo-Albar.stl'):
        self.model_path = model_path
        self.model = None
        self.running = False
        self.latest_frame = None
        self.latest_transform = None
        self.frame_queue = queue.Queue(maxsize=1)  # Queue for thread-safe frame passing
        
        # Fixed camera matrix and distortion for now - replace with your calibrated values if available
        self.camera_matrix = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Load the 3D model
        try:
            self.model = trimesh.load(model_path)
            print(f"Loaded 3D model with {len(self.model.vertices)} vertices")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def capture_thread(self):
        """Thread to continuously capture camera frames and detect ArUco markers"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame.")
                break
                
            # Detect ArUco markers
            result = detect_aruco_markers(frame=frame, debug_display=False)
            
            if result["success"]:
                # Convert rotation vector to rotation matrix
                rvec = result["central_rvec"] 
                tvec = result["central_tvec"]
                rmat, _ = cv2.Rodrigues(rvec)
                
                # Create 4x4 transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rmat
                transform[:3, 3] = tvec.flatten()
                
                # Store the latest transform
                self.latest_transform = transform
                
                # Create a visualization frame
                visual_frame = frame.copy()
                
                # Draw axes at central pose using our camera matrix and dist_coeffs
                cv2.drawFrameAxes(visual_frame, 
                                  self.camera_matrix, 
                                  self.dist_coeffs,
                                  rvec, tvec, 0.1)
                
                # Display distance if available
                if "central_distance" in result:
                    cv2.putText(visual_frame, f"Distance: {result['central_distance']:.3f}m",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Calculate approximate distance from camera
                    distance = np.linalg.norm(tvec)
                    cv2.putText(visual_frame, f"Distance: {distance:.3f}m",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Put the frame in the queue (replaces older frame if queue is full)
                try:
                    # Clear queue before putting new frame
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(visual_frame)
                except queue.Full:
                    pass  # Skip this frame if queue is full
            else:
                # If detection failed, just display the original frame
                try:
                    while not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
                
        cap.release()
        
    def render_scene(self):
        """Set up and render the 3D scene"""
        # Convert trimesh to pyrender mesh
        mesh = pyrender.Mesh.from_trimesh(self.model)
        
        # Create a scene and add the mesh
        scene = pyrender.Scene(bg_color=[0, 0, 0])
        node = scene.add(mesh)
        
        # Add light
        light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
                                  innerConeAngle=np.pi/4, outerConeAngle=np.pi/2)
        scene.add(light, pose=np.eye(4))
        
        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=16.0/9.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)
        
        # Create off-screen renderer for macOS
        if platform.system() == 'Darwin':
            r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
            while self.running:
                if self.latest_transform is not None:
                    scene.set_pose(node, self.latest_transform)
                    color, depth = r.render(scene)
                    # Use OpenCV to display the rendered image
                    color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                    cv2.imshow("3D Model Render", color_rgb)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.running = False
                
                # Process camera frames in the same thread
                try:
                    if not self.frame_queue.empty():
                        frame = self.frame_queue.get_nowait()
                        cv2.imshow("ArUco Detection", frame)
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    
                time.sleep(0.03)  # ~30 FPS
            
            r.delete()
            cv2.destroyAllWindows()
        else:
            # On non-macOS platforms, use the interactive viewer
            viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True)
            
            # Main loop for updating the model's pose
            while self.running:
                if self.latest_transform is not None:
                    viewer.render_lock.acquire()
                    scene.set_pose(node, self.latest_transform)
                    viewer.render_lock.release()
                time.sleep(0.05)
                
            viewer.close_external()
    
    def start(self):
        """Start all the visualization threads"""
        self.running = True
        
        # Start the capture thread
        capture_thread = Thread(target=self.capture_thread)
        capture_thread.daemon = True  # Make thread exit when main program exits
        capture_thread.start()
        
        # Mac-specific handling - run rendering and display on main thread
        if platform.system() == 'Darwin':
            try:
                # Run the 3D rendering and display on the main thread
                self.render_scene()
            except KeyboardInterrupt:
                print("Stopping...")
                self.running = False
            finally:
                self.running = False
                capture_thread.join(timeout=1.0)
                cv2.destroyAllWindows()
        else:
            # On other platforms, run in separate threads
            render_thread = Thread(target=self.render_scene)
            render_thread.daemon = True
            render_thread.start()
            
            # Wait for keyboard interrupt
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("Stopping...")
                self.running = False
            finally:
                self.running = False
                capture_thread.join(timeout=1.0)
                render_thread.join(timeout=1.0)

# The simple_visualization_demo function remains unchanged
def simple_visualization_demo():
    """A simpler version that uses a single frame for testing"""
    # Load the 3D model
    model_path = '/Users/phacharakimpha/comp_vision/ass/aruco_detection/cube/Terrific Borwo-Albar.stl'
    try:
        model = trimesh.load(model_path)
        print(f"Loaded 3D model with {len(model.vertices)} vertices")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Start with a live view for ArUco detection
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
        
    print("Press SPACE to capture a frame for 3D visualization, or q to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow("Press SPACE to capture frame", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):  # Space bar to capture
            result = detect_aruco_markers(frame=frame, debug_display=True)
            
            if result["success"]:
                # Get pose information
                rvec = result["central_rvec"]
                tvec = result["central_tvec"]
                
                # Convert rotation vector to rotation matrix
                rmat, _ = cv2.Rodrigues(rvec)
                
                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = rmat
                transform[:3, 3] = tvec.flatten()
                
                # Apply transformation to model
                model_transformed = model.copy()
                model_transformed.apply_transform(transform)
                
                # Display the model
                print("Showing 3D model. Close window to continue.")
                model_transformed.show()
            else:
                print("ArUco detection failed. Try again.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose one of these methods:
    
    # Method 1: Full interactive visualization
    visualizer = ArUcoModelVisualizer()
    visualizer.start()
    
    # Method 2: Simple demo with capture and view
    # simple_visualization_demo() 