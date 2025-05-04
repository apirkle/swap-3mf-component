import zipfile
import xml.etree.ElementTree as ET
import os
import tempfile
import shutil
from collections import namedtuple
import numpy as np
from stl import mesh

# Constants for 3MF namespace
MODEL_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
PRODUCTION_NAMESPACE = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06"
MODEL_NAMESPACES = {
    "3mf": MODEL_NAMESPACE,
    "p": PRODUCTION_NAMESPACE
}

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

def format_transformation(matrix):
    """Format a transformation matrix into 3MF string format."""
    components = []
    for i in range(3):
        for j in range(4):
            components.append(str(matrix[i][j]))
    return " ".join(components)

def get_component_transform(archive_path, target_component_id):
    """Extract the transformation matrix for a specific component."""
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile:
        raise ValueError("Invalid 3MF file: Not a valid ZIP archive")
        
    # Find the model file
    model_files = [f for f in archive.namelist() if f.endswith(".model")]
    if not model_files:
        raise ValueError("No model file found in 3MF archive")
        
    # Read the main model file
    with archive.open(model_files[0]) as f:
        tree = ET.parse(f)
        root = tree.getroot()
    
    # Find the component in the resources
    resources = root.find("./3mf:resources", MODEL_NAMESPACES)
    if resources is not None:
        for obj in resources.findall("./3mf:object", MODEL_NAMESPACES):
            try:
                obj_id = obj.attrib["id"]
            except KeyError:
                continue
                
            # Look for components in this object
            for comp in obj.findall("./3mf:components/3mf:component", MODEL_NAMESPACES):
                try:
                    comp_id = comp.attrib["objectid"]
                    if comp_id == target_component_id:
                        return parse_transformation(comp.attrib.get("transform", ""))
                except KeyError:
                    continue
    
    raise ValueError(f"Component {target_component_id} not found in 3MF file")

def stl_to_3mf_mesh(stl_path):
    """Convert an STL file to 3MF mesh format."""
    # Read the STL file
    stl_mesh = mesh.Mesh.from_file(stl_path)
    
    # Get unique vertices and create vertex list
    vertices = np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0)
    
    # Create vertex index mapping
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}
    
    # Create triangle list with vertex indices
    triangles = []
    for triangle in stl_mesh.vectors:
        v1 = vertex_map[tuple(triangle[0])]
        v2 = vertex_map[tuple(triangle[1])]
        v3 = vertex_map[tuple(triangle[2])]
        triangles.append((v1, v2, v3))
    
    return vertices.tolist(), triangles

def transform_mesh(stl_mesh):
    """Transform the STL mesh to match the component's coordinate system."""
    # Get the current bounding box
    vertices = stl_mesh.vectors.reshape(-1, 3)
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    center = (min_coords + max_coords) / 2
    
    # Calculate the translation to center the mesh at origin
    translation = -center
    
    # Apply the translation to center the mesh
    vertices += translation
    
    # Create rotation matrix for +90 degrees around X axis
    # [ 1  0   0 ]
    # [ 0  0   1 ]
    # [ 0 -1   0 ]
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])
    
    # Apply rotation
    vertices = np.dot(vertices, rotation_matrix.T)
    
    # Update the mesh vertices
    stl_mesh.vectors = vertices.reshape(-1, 3, 3)
    return stl_mesh

def create_3mf_mesh(stl_mesh):
    """Convert STL mesh data to 3MF mesh format."""
    # Transform the mesh to match the component's coordinate system
    stl_mesh = transform_mesh(stl_mesh)
    
    # Get vertices and create vertex index mapping
    vertices = np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0)
    vertex_map = {tuple(v): i for i, v in enumerate(vertices)}
    
    # Create triangle list with vertex indices
    triangles = []
    for triangle in stl_mesh.vectors:
        v1 = vertex_map[tuple(triangle[0])]
        v2 = vertex_map[tuple(triangle[1])]
        v3 = vertex_map[tuple(triangle[2])]
        triangles.append((v1, v2, v3))
    
    # Create vertices element with namespace
    vertices_element = ET.Element(f"{{{MODEL_NAMESPACE}}}vertices")
    for vertex in vertices:
        vertex_element = ET.SubElement(vertices_element, f"{{{MODEL_NAMESPACE}}}vertex")
        vertex_element.set("x", str(vertex[0]))
        vertex_element.set("y", str(vertex[1]))
        vertex_element.set("z", str(vertex[2]))
    
    # Create triangles element with namespace
    triangles_element = ET.Element(f"{{{MODEL_NAMESPACE}}}triangles")
    for triangle in triangles:
        triangle_element = ET.SubElement(triangles_element, f"{{{MODEL_NAMESPACE}}}triangle")
        triangle_element.set("v1", str(triangle[0]))
        triangle_element.set("v2", str(triangle[1]))
        triangle_element.set("v3", str(triangle[2]))
    
    return vertices_element, triangles_element

def read_stl(stl_path):
    """Read an STL file and return its mesh data."""
    if not os.path.exists(stl_path):
        print(f"STL file {stl_path} does not exist")
        return None
    
    try:
        stl_mesh = mesh.Mesh.from_file(stl_path)
        return stl_mesh
    except Exception as e:
        print(f"Error reading STL file: {e}")
        return None

def swap_component(three_mf_path, stl_path, component_id, output_path):
    """Replace a component in a 3MF file with an STL file."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the 3MF file
        with zipfile.ZipFile(three_mf_path, 'r') as archive:
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
            return False
        
        # Read the STL file
        stl_mesh = read_stl(stl_path)
        if stl_mesh is None:
            return False
        
        # Convert STL mesh to 3MF format
        vertices_element, triangles_element = create_3mf_mesh(stl_mesh)
        
        # Find the object file for the component
        object_path = None
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file == f'Object_{component_id}_1.model':
                    object_path = os.path.join(root, file)
                    break
            if object_path:
                break
        
        if not object_path:
            print(f"Object file for component {component_id} not found")
            return False
        
        # Read and parse the original file to preserve namespaces
        with open(object_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Get the original XML declaration
            xml_decl = ''
            if content.startswith('<?xml'):
                xml_decl = content[:content.find('?>') + 2] + '\n'
            
            # Parse the XML
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            object_tree = ET.fromstring(content, parser=parser)
            
            # Register namespaces with their original prefixes
            namespaces = {
                '': "http://schemas.microsoft.com/3dmanufacturing/core/2015/02",
                'p': "http://schemas.microsoft.com/3dmanufacturing/production/2015/06",
                'slic3rpe': "http://schemas.slic3r.org/3mf/2017/06"
            }
            for key, value in namespaces.items():
                if key:
                    ET.register_namespace(key, value)
                else:
                    ET.register_namespace('', value)
        
        # Find or create mesh element
        mesh_element = object_tree.find(f".//{{{MODEL_NAMESPACE}}}mesh")
        if mesh_element is None:
            mesh_element = ET.SubElement(object_tree, f"{{{MODEL_NAMESPACE}}}mesh")
        else:
            # Clear existing vertices and triangles
            vertices_element_old = mesh_element.find(f".//{{{MODEL_NAMESPACE}}}vertices")
            if vertices_element_old is not None:
                mesh_element.remove(vertices_element_old)
            triangles_element_old = mesh_element.find(f".//{{{MODEL_NAMESPACE}}}triangles")
            if triangles_element_old is not None:
                mesh_element.remove(triangles_element_old)
        
        # Add new mesh data
        mesh_element.append(vertices_element)
        mesh_element.append(triangles_element)
        
        # Save the updated object file while preserving the XML format
        with open(object_path, 'w', encoding='utf-8') as f:
            f.write(xml_decl)
            f.write(ET.tostring(object_tree, encoding='unicode', xml_declaration=False))
        
        # Create a new 3MF file
        with zipfile.ZipFile(output_path, 'w') as new_archive:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_archive.write(file_path, arcname)
        
        print(f"Successfully created modified 3MF file at {output_path}")
        return True

def add_component(three_mf_path, stl_path, output_path):
    """Add a new component to a 3MF file."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the 3MF file
        with zipfile.ZipFile(three_mf_path, 'r') as archive:
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
            return False
        
        # Read the STL file
        stl_mesh = read_stl(stl_path)
        if stl_mesh is None:
            return False
        
        # Convert STL mesh to 3MF format with the same transformations
        vertices_element, triangles_element = create_3mf_mesh(stl_mesh)
        
        # Read and parse the model file to preserve namespaces
        with open(model_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Get the original XML declaration
            xml_decl = ''
            if content.startswith('<?xml'):
                xml_decl = content[:content.find('?>') + 2] + '\n'
            
            # Parse the XML
            parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
            model_tree = ET.fromstring(content, parser=parser)
            
            # Register namespaces
            namespaces = {
                '': MODEL_NAMESPACE,
                'p': PRODUCTION_NAMESPACE
            }
            for key, value in namespaces.items():
                if key:
                    ET.register_namespace(key, value)
                else:
                    ET.register_namespace('', value)
        
        # Find or create resources element
        resources = model_tree.find(f".//{{{MODEL_NAMESPACE}}}resources")
        if resources is None:
            resources = ET.SubElement(model_tree, f"{{{MODEL_NAMESPACE}}}resources")
        
        # Create new object with unique ID
        object_id = 1
        while True:
            existing_object = resources.find(f".//{{{MODEL_NAMESPACE}}}object[@id='{object_id}']")
            if existing_object is None:
                break
            object_id += 1
        
        # Create new object element
        new_object = ET.SubElement(resources, f"{{{MODEL_NAMESPACE}}}object")
        new_object.set("id", str(object_id))
        new_object.set("type", "model")
        
        # Add mesh data to the new object
        mesh_element = ET.SubElement(new_object, f"{{{MODEL_NAMESPACE}}}mesh")
        mesh_element.append(vertices_element)
        mesh_element.append(triangles_element)
        
        # Find or create build element
        build = model_tree.find(f".//{{{MODEL_NAMESPACE}}}build")
        if build is None:
            build = ET.SubElement(model_tree, f"{{{MODEL_NAMESPACE}}}build")
        
        # Add new build item
        build_item = ET.SubElement(build, f"{{{MODEL_NAMESPACE}}}item")
        build_item.set("objectid", str(object_id))
        
        # Save the updated model file
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(xml_decl)
            f.write(ET.tostring(model_tree, encoding='unicode', xml_declaration=False))
        
        # Create a new 3MF file
        with zipfile.ZipFile(output_path, 'w') as new_archive:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    new_archive.write(file_path, arcname)
        
        print(f"Successfully created modified 3MF file at {output_path}")
        return True

def main():
    import sys
    
    if len(sys.argv) < 4:
        print("Usage:")
        print("To swap a component: python swap_3mf_component.py <original_3mf> <new_stl> <target_component_id> <output_3mf>")
        print("To add a component: python swap_3mf_component.py <original_3mf> <new_stl> add <output_3mf>")
        sys.exit(1)
        
    try:
        if sys.argv[3].lower() == 'add':
            add_component(sys.argv[1], sys.argv[2], sys.argv[4])
        else:
            swap_component(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 