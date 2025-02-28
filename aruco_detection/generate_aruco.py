import cv2
import numpy as np

# Define the dictionary of supported ArUco marker types
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}

def generate_aruco_marker(id, dict_type="DICT_7X7_1000", size=200):
    """
    Generate an ArUco marker image
    
    Args:
        id: Marker ID to generate
        dict_type: ArUco dictionary type
        size: Size of the output marker image in pixels
        
    Returns:
        Marker image (numpy array)
    """
    if dict_type not in ARUCO_DICT:
        print(f"ArUco type {dict_type} not supported. Using DICT_7X7_1000 instead.")
        dict_type = "DICT_7X7_1000"
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, id, size)
    return marker_img

def extract_binary_pattern(marker_id, dict_type="DICT_7X7_1000"):
    """
    Extract the binary pattern from an ArUco marker
    
    Args:
        marker_id: ID of the marker
        dict_type: Dictionary type
        
    Returns:
        Binary pattern as an integer
    """
    # Generate a small image of the marker
    marker_img = generate_aruco_marker(marker_id, dict_type, 100)
    
    # Determine grid size from dictionary type
    if "7X7" in dict_type:
        grid_size = 7
    elif "6X6" in dict_type:
        grid_size = 6
    elif "5X5" in dict_type:
        grid_size = 5
    elif "4X4" in dict_type:
        grid_size = 4
    else:
        grid_size = 7
    
    # The binary pattern is stored in the inner part, excluding the border
    cell_size = marker_img.shape[0] // grid_size
    
    # Extract the inner grid (excluding the border)
    binary_pattern = 0
    for i in range(1, grid_size-1):
        for j in range(1, grid_size-1):
            cell_center_y = int((i + 0.5) * cell_size)
            cell_center_x = int((j + 0.5) * cell_size)
            
            # Black is 1, white is 0 (for ArUco markers)
            bit = 1 if marker_img[cell_center_y, cell_center_x] == 0 else 0
            binary_pattern = (binary_pattern << 1) | bit
    
    return binary_pattern

def generate_binary_patterns(dict_type="DICT_7X7_1000", max_ids=50):
    """
    Generate a dictionary of binary patterns for ArUco markers
    
    Args:
        dict_type: Dictionary type
        max_ids: Maximum number of IDs to generate (to avoid excessive computation)
        
    Returns:
        Dictionary mapping marker IDs to their binary patterns
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[dict_type])
    max_markers = aruco_dict.bytesList.shape[0]
    
    # Limit the number of markers to generate
    num_markers = min(max_markers, max_ids)
    
    binary_patterns = {}
    for id in range(num_markers):
        binary_patterns[id] = extract_binary_pattern(id, dict_type)
    
    return binary_patterns

# Pre-compute binary patterns for common ArUco dictionaries
# These can be imported and used in detection code
ARUCO_BINARY_PATTERNS = {
    "DICT_7X7_1000": generate_binary_patterns("DICT_7X7_1000", 50),
    "DICT_6X6_1000": generate_binary_patterns("DICT_6X6_1000", 50),
    "DICT_5X5_1000": generate_binary_patterns("DICT_5X5_1000", 50),
    "DICT_4X4_1000": generate_binary_patterns("DICT_4X4_1000", 50)
}

def save_marker(id, dict_type="DICT_7X7_1000", size=200, filename=None):
    """
    Generate and save an ArUco marker to a file
    
    Args:
        id: Marker ID
        dict_type: ArUco dictionary type
        size: Size in pixels
        filename: Output filename, if None will use format "aruco_<dict_type>_<id>.png"
    """
    marker_img = generate_aruco_marker(id, dict_type, size)
    
    if filename is None:
        filename = f"aruco_{dict_type}_{id}.png"
    
    cv2.imwrite(filename, marker_img)
    print(f"Saved marker {id} from {dict_type} to {filename}")

if __name__ == "__main__":
    # Example usage: generate and save markers
    for id in range(5):
        save_marker(id, "DICT_7X7_1000", 300)
    
    # Print binary patterns for verification
    patterns = generate_binary_patterns("DICT_7X7_1000", 5)
    for id, pattern in patterns.items():
        print(f"Marker ID {id}: {bin(pattern)}")

