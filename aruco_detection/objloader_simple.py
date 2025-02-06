import numpy as np

def load_obj(filename):
    """
    Returns a dict with:
      - 'vertices': list of [x, y, z]
      - 'faces': list of face indices [[v1, v2, v3], ...] or quads
    """
    vertices = []
    faces = []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] == 'v':  # Vertex
                # e.g. "v 1.0 2.0 3.0"
                x, y, z = map(float, parts[1:4])
                vertices.append([x, y, z])

            elif parts[0] == 'f':  # Face
                # e.g. "f 1 2 3" or "f 1/1 2/2 3/3" etc.
                # We'll just split on '/' and take the first number for vertex index:
                idxs = []
                for vtx in parts[1:]:
                    v = vtx.split('/')[0]  # handle "f v/t/n"
                    idxs.append(int(v) - 1)  # OBJ indices are 1-based
                faces.append(idxs)

    return {
        'vertices': np.array(vertices, dtype=np.float32),
        'faces': faces
    }
