import zipfile
from lxml import etree as ET
import os
import numpy as np
from stl import mesh
from collections import namedtuple
import trimesh
import sys
import tempfile

# Constants for 3MF namespace
MODEL_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PRODUCTION_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
MODEL_NAMESPACES = {
    "3mf": MODEL_NAMESPACE,
    "p": PRODUCTION_NAMESPACE
}

# Data structures
MeshObject = namedtuple("MeshObject", ["id", "name", "vertices", "triangles", "metadata", "transform", "components", "bbox", "volume"])
Component = namedtuple("Component", ["object_id", "transform", "bbox", "volume"])

class Matrix4x4:
    """Simple 4x4 matrix implementation for transformations."""
    def __init__(self):
        self.matrix = [[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]]
    
    def __getitem__(self, index):
        return self.matrix[index]
    
    def __setitem__(self, index, value):
        self.matrix[index] = value
    
    def __str__(self):
        return "\n".join([f"[{row[0]:.3f}, {row[1]:.3f}, {row[2]:.3f}, {row[3]:.3f}]" for row in self.matrix])

def identity_matrix():
    """Create an identity matrix."""
    return Matrix4x4()

def parse_transformation(transform_str):
    """Parse a transformation matrix from 3MF format string."""
    if not transform_str:
        return identity_matrix()
    
    components = transform_str.split()
    matrix = Matrix4x4()
    
    if len(components) != 12:
        return matrix
        
    # 3MF uses row-major format
    for i in range(3):
        for j in range(4):
            try:
                matrix[i][j] = float(components[i*4 + j])
            except (ValueError, IndexError):
                pass
                
    return matrix

def read_vertices(vertex_elements):
    """Read vertices from XML elements."""
    vertices = []
    for vertex in vertex_elements:
        try:
            x = float(vertex.attrib.get("x", 0))
            y = float(vertex.attrib.get("y", 0))
            z = float(vertex.attrib.get("z", 0))
            vertices.append([x, y, z])
        except ValueError:
            continue
    return vertices

def read_triangles(triangle_elements):
    """Read triangles from XML elements."""
    triangles = []
    for triangle in triangle_elements:
        try:
            v1 = int(triangle.attrib["v1"])
            v2 = int(triangle.attrib["v2"])
            v3 = int(triangle.attrib["v3"])
            triangles.append([v1, v2, v3])
        except (KeyError, ValueError):
            continue
    return triangles

def read_metadata(metadata_elements):
    """Read metadata from XML elements."""
    metadata = {}
    for meta in metadata_elements:
        name = meta.attrib.get("name")
        if name:
            metadata[name] = meta.text
    return metadata

def calculate_bbox_and_volume(vertices, triangles):
    """Calculate bounding box and volume for a mesh."""
    if not vertices or not triangles:
        return None, 0.0
    
    # Convert to numpy arrays for easier calculations
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    
    # Calculate bounding box
    bbox = {
        'min': np.min(vertices, axis=0),
        'max': np.max(vertices, axis=0),
        'size': np.max(vertices, axis=0) - np.min(vertices, axis=0)
    }
    
    # Calculate volume using the shoelace formula
    volume = 0.0
    for tri in triangles:
        v1, v2, v3 = vertices[tri]
        volume += np.dot(v1, np.cross(v2, v3)) / 6.0
    
    return bbox, abs(volume)

def analyze_stl(stl_path):
    """Analyze an STL file and return its properties."""
    try:
        # Read the STL file
        stl_mesh = mesh.Mesh.from_file(stl_path)
        
        # Get vertices and triangles
        vertices = stl_mesh.vectors.reshape(-1, 3)
        triangles = np.arange(len(vertices)).reshape(-1, 3)
        
        # Calculate bounding box
        min_coords = np.min(vertices, axis=0)
        max_coords = np.max(vertices, axis=0)
        dimensions = max_coords - min_coords
        
        # Calculate volume using trimesh
        trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        volume = trimesh_mesh.volume
        
        return {
            'vertices': vertices,
            'triangles': triangles,
            'vertex_count': len(vertices),
            'triangle_count': len(triangles),
            'bounding_box': {
                'min': min_coords,
                'max': max_coords,
                'dimensions': dimensions
            },
            'volume': volume
        }
    except Exception as e:
        print(f"Error analyzing STL file: {str(e)}")
        return None

def calculate_bbox(vertices):
    if not vertices:
        return None
    
    # Initialize min and max coordinates with the first vertex
    min_x = max_x = vertices[0][0]
    min_y = max_y = vertices[0][1]
    min_z = max_z = vertices[0][2]
    
    # Find min and max coordinates
    for x, y, z in vertices:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
        min_z = min(min_z, z)
        max_z = max(max_z, z)
    
    return {
        'min': (min_x, min_y, min_z),
        'max': (max_x, max_y, max_z),
        'dimensions': (max_x - min_x, max_y - min_y, max_z - min_z)
    }

def calculate_volume(vertices, triangles):
    if not vertices or not triangles:
        return 0.0
    
    volume = 0.0
    for v1_idx, v2_idx, v3_idx in triangles:
        # Get the vertices of the triangle
        v1 = vertices[v1_idx]
        v2 = vertices[v2_idx]
        v3 = vertices[v3_idx]
        
        # Calculate the signed volume of the tetrahedron formed by the triangle and the origin
        x1, y1, z1 = v1
        x2, y2, z2 = v2
        x3, y3, z3 = v3
        
        # Calculate the volume using the determinant formula
        volume += (1.0/6.0) * (
            -x3*y2*z1 + x2*y3*z1 + x3*y1*z2 - x1*y3*z2 - x2*y1*z3 + x1*y2*z3
        )
    
    return abs(volume)

def read_mesh_data(mesh_element, namespaces):
    if mesh_element is None:
        return [], []
    
    vertices = []
    triangles = []
    
    # Find vertices
    vertices_element = mesh_element.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}vertices')
    if vertices_element is None:
        vertices_element = mesh_element.find('vertices')
    
    if vertices_element is not None:
        for vertex in vertices_element.findall('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}vertex'):
            if vertex is None:
                continue
            x = float(vertex.get('x', 0))
            y = float(vertex.get('y', 0))
            z = float(vertex.get('z', 0))
            vertices.append((x, y, z))
    
    # Find triangles
    triangles_element = mesh_element.find('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangles')
    if triangles_element is None:
        triangles_element = mesh_element.find('triangles')
    
    if triangles_element is not None:
        for triangle in triangles_element.findall('{http://schemas.microsoft.com/3dmanufacturing/core/2015/02}triangle'):
            if triangle is None:
                continue
            v1 = int(triangle.get('v1', 0))
            v2 = int(triangle.get('v2', 0))
            v3 = int(triangle.get('v3', 0))
            triangles.append((v1, v2, v3))
    
    return vertices, triangles

def analyze_3mf_file(file_path):
    """Analyze a 3MF file and return its components and build items."""
    try:
        # Extract the 3MF file
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(file_path, 'r') as archive:
                archive.extractall(temp_dir)
            
            # Find the model file
            model_path = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('3dmodel.model'):
                        model_path = os.path.join(root, file)
                        break
                if model_path:
                    break
            
            if not model_path:
                print("No model file found in 3MF archive")
                return None, None, None
            
            # Parse the XML with namespaces
            parser = ET.XMLParser(remove_blank_text=True)
            tree = ET.parse(model_path, parser)
            root = tree.getroot()
            
            # Get namespaces
            namespaces = {'': "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
            for key, value in root.nsmap.items():
                if key is None:
                    namespaces[''] = value
                else:
                    namespaces[key] = value
            
            # Find all objects
            objects = {}
            for obj in root.findall('.//object', namespaces):
                obj_id = obj.get('id')
                if obj_id:
                    mesh = obj.find('.//mesh', namespaces)
                    if mesh is not None:
                        vertices = []
                        triangles = []
                        
                        # Get vertices
                        for vertex in mesh.findall('.//vertex', namespaces):
                            x = float(vertex.get('x', 0))
                            y = float(vertex.get('y', 0))
                            z = float(vertex.get('z', 0))
                            vertices.append((x, y, z))
                        
                        # Get triangles
                        for triangle in mesh.findall('.//triangle', namespaces):
                            v1 = int(triangle.get('v1', 0))
                            v2 = int(triangle.get('v2', 0))
                            v3 = int(triangle.get('v3', 0))
                            triangles.append((v1, v2, v3))
                        
                        # Calculate bounding box and volume
                        bbox = calculate_bbox(vertices)
                        volume = calculate_volume(vertices, triangles)
                        
                        objects[obj_id] = {
                            'vertices': vertices,
                            'triangles': triangles,
                            'vertex_count': len(vertices),
                            'triangle_count': len(triangles),
                            'bounding_box': bbox,
                            'volume': volume
                        }
            
            # Find all components
            components = []
            for comp in root.findall('.//component', namespaces):
                obj_id = comp.get('objectid')
                if obj_id:
                    transform_str = comp.get('transform')
                    if transform_str:
                        transform = [float(x) for x in transform_str.split()]
                    else:
                        transform = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # Identity matrix
                    
                    components.append({
                        'object_id': obj_id,
                        'transform': transform
                    })
            
            # Find all build items
            build_items = []
            for item in root.findall('.//item', namespaces):
                obj_id = item.get('objectid')
                if obj_id:
                    transform_str = item.get('transform')
                    if transform_str:
                        transform = [float(x) for x in transform_str.split()]
                    else:
                        transform = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]  # Identity matrix
                    
                    build_items.append({
                        'object_id': obj_id,
                        'transform': transform
                    })
            
            return objects, components, build_items
    
    except Exception as e:
        print(f"Error analyzing 3MF file: {str(e)}")
        return None, None, None

def print_analysis(objects, components, build_items):
    print("\n=== 3MF Analysis ===\n")
    
    if objects:
        print("Objects:")
        for obj in objects:
            print(f"\nObject {obj['id']}:")
            print(f"  Vertices: {obj['vertex_count']}")
            print(f"  Triangles: {obj['triangle_count']}")
            if obj['bounding_box']:
                print(f"  Bounding Box:")
                print(f"    Min: ({obj['bounding_box']['min'][0]:.3f}, {obj['bounding_box']['min'][1]:.3f}, {obj['bounding_box']['min'][2]:.3f})")
                print(f"    Max: ({obj['bounding_box']['max'][0]:.3f}, {obj['bounding_box']['max'][1]:.3f}, {obj['bounding_box']['max'][2]:.3f})")
                print(f"    Dimensions: ({obj['bounding_box']['dimensions'][0]:.3f}, {obj['bounding_box']['dimensions'][1]:.3f}, {obj['bounding_box']['dimensions'][2]:.3f})")
            print(f"  Volume: {obj['volume']:.2f} cubic units")
    
    if components:
        print("\nComponents:")
        for comp in components:
            print(f"\nComponent (Object {comp['object_id']}):")
            print(f"  Vertices: {comp['vertex_count']}")
            print(f"  Triangles: {comp['triangle_count']}")
            if comp['bounding_box']:
                print(f"  Bounding Box:")
                print(f"    Min: ({comp['bounding_box']['min'][0]:.3f}, {comp['bounding_box']['min'][1]:.3f}, {comp['bounding_box']['min'][2]:.3f})")
                print(f"    Max: ({comp['bounding_box']['max'][0]:.3f}, {comp['bounding_box']['max'][1]:.3f}, {comp['bounding_box']['max'][2]:.3f})")
                print(f"    Dimensions: ({comp['bounding_box']['dimensions'][0]:.3f}, {comp['bounding_box']['dimensions'][1]:.3f}, {comp['bounding_box']['dimensions'][2]:.3f})")
            print(f"  Volume: {comp['volume']:.2f} cubic units")
            if comp['transform']:
                # Extract translation from transformation matrix
                tx = comp['transform'][9] if len(comp['transform']) > 9 else 0
                ty = comp['transform'][10] if len(comp['transform']) > 10 else 0
                tz = comp['transform'][11] if len(comp['transform']) > 11 else 0
                print(f"  Translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")
    
    if build_items:
        print("\nBuild Items:")
        for item in build_items:
            print(f"\nInstance of Object {item['object_id']}:")
            if item['transform']:
                # Extract translation from transformation matrix
                tx = item['transform'][9] if len(item['transform']) > 9 else 0
                ty = item['transform'][10] if len(item['transform']) > 10 else 0
                tz = item['transform'][11] if len(item['transform']) > 11 else 0
                print(f"  Translation: ({tx:.3f}, {ty:.3f}, {tz:.3f})")

def find_matching_component(stl_data, components, tolerance=0.1):
    """Find the component that best matches the STL file's dimensions."""
    if not stl_data or not components:
        return None
    
    stl_bbox = stl_data['bounding_box']
    stl_dims = stl_bbox['dimensions']
    stl_volume = stl_data['volume']
    
    best_match = None
    best_score = float('inf')
    
    for comp in components:
        comp_bbox = comp['bounding_box']
        comp_dims = comp_bbox['dimensions']
        comp_volume = comp['volume']
        
        # Calculate relative differences in dimensions and volume
        dim_diff = sum(abs(s - c) / max(s, c) for s, c in zip(stl_dims, comp_dims)) / 3
        vol_diff = abs(stl_volume - comp_volume) / max(stl_volume, comp_volume)
        
        # Combined score (lower is better)
        score = (dim_diff + vol_diff) / 2
        
        if score < best_score and score < tolerance:
            best_score = score
            best_match = comp
    
    return best_match

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_3mf.py <3mf_file> [stl_file]")
        return
    
    file_path = sys.argv[1]
    
    if file_path.endswith('.stl'):
        # Analyze STL file
        print("\n=== STL Analysis ===\n")
        stl_data = analyze_stl(file_path)
        if stl_data:
            print(f"Vertices: {stl_data['vertex_count']}")
            print(f"Triangles: {stl_data['triangle_count']}")
            print("Bounding Box:")
            print(f"  Min: {tuple(stl_data['bounding_box']['min'])}")
            print(f"  Max: {tuple(stl_data['bounding_box']['max'])}")
            print(f"  Dimensions: {tuple(stl_data['bounding_box']['dimensions'])}")
            print(f"Volume: {stl_data['volume']} cubic units")
    else:
        # Analyze 3MF file
        print("\n=== 3MF Analysis ===\n")
        try:
            objects, components, build_items = analyze_3mf_file(file_path)
            if objects:
                print("\nComponents:\n")
                for obj_id, obj_data in objects.items():
                    print(f"Component (Object {obj_id}):")
                    print(f"  Vertices: {obj_data['vertex_count']}")
                    print(f"  Triangles: {obj_data['triangle_count']}")
                    print("  Bounding Box:")
                    print(f"    Min: {obj_data['bounding_box']['min']}")
                    print(f"    Max: {obj_data['bounding_box']['max']}")
                    print(f"    Dimensions: {obj_data['bounding_box']['dimensions']}")
                    print(f"  Volume: {obj_data['volume']:.2f} cubic units")
                    
                    # Find corresponding component
                    for comp in components:
                        if comp['object_id'] == obj_id:
                            print(f"  Translation: ({comp['transform'][9]:.3f}, {comp['transform'][10]:.3f}, {comp['transform'][11]:.3f})")
                            break
                    print()
            
            if build_items:
                print("\nBuild Items:\n")
                for item in build_items:
                    print(f"Instance of Object {item['object_id']}:")
                    print(f"  Translation: ({item['transform'][9]:.3f}, {item['transform'][10]:.3f}, {item['transform'][11]:.3f})")
                    print()
        except Exception as e:
            print(f"Error analyzing 3MF file: {str(e)}")
            return

if __name__ == "__main__":
    main() 