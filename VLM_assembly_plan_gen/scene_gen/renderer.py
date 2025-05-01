import bpy
import os
import math
import numpy as np
import json
import mathutils
from mathutils import Vector, Matrix
import pickle
import argparse
import random
import sys
from bpy_extras.object_utils import world_to_camera_view

engine = "BLENDER_EEVEE"  # for better light effect
context = bpy.context
scene = context.scene
render = scene.render
bpy.context.scene.view_settings.exposure = 1.0

render.engine = engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 1024
render.resolution_y = 1024
render.resolution_percentage = 100

scene.cycles.device = "GPU"  # “GPU”
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

parser = argparse.ArgumentParser()
parser.add_argument('--random_translate', type=str)
parser.add_argument('--random_rotate', type=str)
parser.add_argument('--input_data_dir', type=str)
parser.add_argument('--iteration_index', type=str)
parser.add_argument('--json_path', type=str)
parser.add_argument('--keypoints_data_path', type=str)
parser.add_argument('--output_data_dir', type=str)
parser.add_argument('--output_file_name', type=str)
parser.add_argument('--blend_file_path', type=str)
parser.add_argument("--no_3d_keypoints", action="store_true", default=False) 
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])

def apply_texture_to_object(obj, color=(0.0, 0.0, 0.0, 1.0)):  # Default black
    """
    Applies a solid color texture to the given object.

    :param obj: The object to apply the texture to.
    :param color: The RGBA color of the texture (default black).
    """
    # Check if the object has geometry data
    if obj.data is None:
        print(f"Object {obj.name} has no data. Skipping texture.")
        return

    # Create a new material with the given color
    material_name = f"Material_{color}"
    if material_name not in bpy.data.materials:
        material = bpy.data.materials.new(name=material_name)
        material.use_nodes = True
        nodes = material.node_tree.nodes
        bsdf = nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = color  # RGBA color
    else:
        material = bpy.data.materials[material_name]

    # Assign the material to the object
    if not obj.data.materials:
        obj.data.materials.append(material)
    else:
        obj.data.materials[0] = material

    print(f"Applied material {material_name} to {obj.name}.")


def setup_environment_lighting(location=(0, 0, 2), energy=10, color=(1.0, 1.0, 1.0)):
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name="light_up", type='POINT')
    light_data.energy = energy
    light_data.color = color

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name="light_2.80", object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object

    # change location
    light_object.location = location

    print("[INFO] Successfully set up light.")


def bounding_box_for_objects(objects):
    """
    Returns:
        (center, radius)
        where 'center' is a Vector with (x,y,z),
        and 'radius' is the bounding-sphere radius.
    """
    if not objects:
        return Vector((0, 0, 0)), 0.0

    # Initialize min/max to extreme values
    min_x = min_y = min_z = float('inf')
    max_x = max_y = max_z = float('-inf')

    # Iterate over every object's bounding-box corners in world coords
    for obj in objects:
        for local_corner in obj.bound_box:  # corners in local space
            world_corner = obj.matrix_world @ Vector(local_corner)
            x, y, z = world_corner
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    # Center is midpoint of min/max
    center = Vector((
        (min_x + max_x) / 2.0,
        (min_y + max_y) / 2.0,
        (min_z + max_z) / 2.0
    ))

    # Diagonal length of the bounding box
    dx = (max_x - min_x)
    dy = (max_y - min_y)
    dz = (max_z - min_z)
    diag_length = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Radius is half the diagonal
    radius = diag_length / 2.0

    return center, radius


def look_at(camera, target):
    """
    Orient 'camera' so that it points at 'target'.
    """
    direction = target - camera.location
    direction.normalize()
    rot_quat = direction.to_track_quat('-Z', 'Y')  # Blender uses -Z as forward, +Y as up
    camera.rotation_euler = rot_quat.to_euler()


def project_3d_to_2d(scene, camera, keypoint_coords):
    """
    Project a 3D coordinate to 2D using the active camera.
    :param scene: The active Blender scene.
    :param camera: The camera object.
    :param keypoint_coords: A list of dictionaries containing 3D keypoints with IDs.
    :return: A list of dictionaries with 2D coordinates and IDs.
    """
    projected_keypoints = []
    print(keypoint_coords)
    for keypoint in keypoint_coords:
        coords_3d = Vector(keypoint['location'])
        coords_2d = world_to_camera_view(scene, camera, coords_3d)
        print(f"Normalized: {coords_2d.x:.3f}, {coords_2d.y:.3f}, {coords_2d.z:.3f}")
        # Convert to pixel coordinates
        x = round(coords_2d.x * scene.render.resolution_x)
        y = round(scene.render.resolution_y - (coords_2d.y * scene.render.resolution_y))
        projected_keypoints.append({
            # 'ID': keypoint['ID'],
            'name': keypoint['name'],
            'coords_2d': (x, y)
        })
    print(projected_keypoints)
    return projected_keypoints


def rotate_camera_around_objects(
        objects,
        keypoints,
        latitudes_degs=(88),
        num_longitudes=1,
        margin_factor=2.5,
):
    """
    Rotate the camera around the objects, taking snapshots at the specified latitudes
    and equally spaced longitudes. Also projects 3D keypoints to 2D.
    """
    if not objects:
        print("No objects provided for camera rotation.")
        return

    if not keypoints:
        print("No keypoints provided for projection.")
        return

    bpy.context.view_layer.update()

    # Compute the bounding box center & bounding-sphere radius
    center, base_radius = bounding_box_for_objects(objects)
    if base_radius < 1e-3:
        base_radius = 1.0

    # Expand the radius slightly to ensure we see everything comfortably
    radius = base_radius * margin_factor

    print(f"Camera center: {center}, radius (with margin): {radius}")

    # Create a camera object if one doesn’t exist
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    frame_idx = 0
    scene = bpy.context.scene

    # For each latitude in degrees ...
    # for lat_deg in latitudes_degs:
    phi = math.radians(90 - 88)
    for j in range(num_longitudes):
        theta = (2.0 * math.pi) * (j / num_longitudes)

        # Convert spherical → Cartesian
        x = center.x + radius * math.sin(phi) * math.cos(theta)
        y = center.y + radius * math.sin(phi) * math.sin(theta)
        z = center.z + radius * math.cos(phi)

        # Move camera
        camera.location = (x, y, z)
        look_at(camera, center)
        bpy.context.view_layer.update()

        # Render
        output_image = os.path.join(output_dir, output_file)
        bpy.context.scene.render.filepath = output_image
        print(f"Rendering frame {frame_idx} (lat={88}, lon={j}) to {output_image}")
        bpy.ops.render.render(write_still=True)

        # Project 3D keypoints to 2D
        projected_keypoints = project_3d_to_2d(scene, camera, keypoints)
        print("====================================")
        print(projected_keypoints)
        print("====================================")
        # Save projected keypoints to a file for each frame using Pickle
        projection_output = os.path.join(output_dir, f"scene_projections.pkl")
        with open(projection_output, "wb") as f:
            pickle.dump(projected_keypoints, f)



def direction_to_quaternion(direction):
    direction = direction / np.linalg.norm(direction)
    z_axis = np.array([0, 0, 1])
    axis = np.cross(z_axis, direction)
    angle = math.acos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))  # Ensure value is in valid range
    if np.linalg.norm(axis) < 1e-6:  # Handle the case when axis is zero (direction aligns with z-axis)
        axis = np.array([1.0, 0.0, 0.0])  # Use an arbitrary perpendicular axis
    else:
        axis = axis / np.linalg.norm(axis)  # Normalize the axis
    quaternion = mathutils.Quaternion(axis.tolist(), angle)
    return quaternion


# Function to clear all objects in the scene
def clear_scene():
    print("Clearing the scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    print("Scene cleared.")


# Helper to load and position a single OBJ file
def load_obj(filepath, position=None):
    print(f"Loading OBJ file: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    bpy.ops.import_scene.obj(filepath=filepath)
    obj = bpy.context.selected_objects[0]  # Get the last imported object

    # Ensure position is 3D
    if position is not None:
        if len(position) == 2:  # Add a default z-coordinate if missing
            position = (position[0], position[1], 0.0)
        obj.location = position
        print(f"Set object location to {position}")
    else:
        print("Using original object coordinates.")
    return obj


# Helper to create a random color material
def create_material_with_color(name, color):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)  # RGBA, Alpha=1
    return material


# Helper to create a keypoint at a specific location with a unique color
def create_keypoint(name, coords):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.05, location=coords)
    keypoint = bpy.context.object
    keypoint.name = name

    # Generate a random color for the keypoint
    random_color = (1, 0, 0)  # RGB values between 0 and 1
    material_name = f"{name}_Material"
    material = create_material_with_color(material_name, random_color)

    # Assign the material to the keypoint
    if not any(mat.name == material_name for mat in keypoint.data.materials):  # Check by material name
        keypoint.data.materials.append(material)
    return keypoint


# Helper to merge keypoints and objects
def parent_keypoints_to_object(obj, keypoints):
    for kp in keypoints:
        keypoint = create_keypoint(kp["name"], kp["coords"])
        keypoint.parent = obj  # Parent the keypoint to the object


# Helper to merge multiple objects into one
def merge_objects(objects):
    if not objects:
        return None
    bpy.context.view_layer.objects.active = objects[0]
    for obj in objects:
        obj.select_set(True)
    bpy.ops.object.join()  # Join the selected objects into one
    return objects[0]  # Return the merged object


def filter_keypoints(merged_keypoints):
    seen_once = set()  # Names encountered once
    duplicates = set()  # Names encountered more than once

    # First pass: Identify duplicates
    for sublist in merged_keypoints:
        for keypoint in sublist:
            name = keypoint["name"]
            if name in seen_once:
                duplicates.add(name)
            else:
                seen_once.add(name)

    for i in range(len(merged_keypoints)):
        merged_keypoints[i] = [kp for kp in merged_keypoints[i] if kp["name"] not in duplicates]

    return merged_keypoints


def group_keypoints_and_save(input_keypoints, output_file):
    from collections import defaultdict

    # Group keypoints by name
    grouped = defaultdict(list)
    for keypoint in input_keypoints:
        grouped[keypoint["name"]].append(keypoint["ID"])

    # Collect pairs (names with exactly 2 IDs)
    pairs = [(ids[0], ids[1]) for ids in grouped.values() if len(ids) == 2]

    # Convert those pairs to a JSON-friendly list
    output_list = []
    for pair in pairs:
        output_list.append({
            "connection point 1": pair[0],
            "connection point 2": pair[1],
        })

    # Write to file in JSON format
    with open(output_file, 'w') as f:
        json.dump(output_list, f, indent=4)


def align_largest_area_on_xy(obj, rand_rotate):
    """
    Rotates `obj` so that the face of its bounding box with the largest area
    becomes parallel to the XY plane (i.e., its normal aligns with +Z).
    """
    # 1) Apply the object's transforms so bounding_box is 'clean'
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 2) Get the bounding-box corners in local space (not world space)
    #    obj.bound_box gives 8 corners in *local* coordinates, from min to max.
    bbox_corners_local = [Vector(corner) for corner in obj.bound_box]

    # Helper to compute area given 4 corner indices of a face
    def face_area(idx0, idx1, idx2, idx3):
        v1 = bbox_corners_local[idx1] - bbox_corners_local[idx0]
        v2 = bbox_corners_local[idx2] - bbox_corners_local[idx0]
        return v1.cross(v2).length

    # Each tuple lists 4 corner indices describing a face of the bounding box
    faces = [
        (0, 1, 2, 3),  # Bottom
        (4, 5, 6, 7),  # Top
        (0, 1, 5, 4),  # Front
        (2, 3, 7, 6),  # Back
        (0, 3, 7, 4),  # Left
        (1, 2, 6, 5),  # Right
    ]

    # 3) Find the face with the largest area
    max_area = 0.0
    largest_face = None
    for face in faces:
        area = face_area(*face)
        if area > max_area:
            max_area = area
            largest_face = face

    # Compute the local-space normal of that face
    i0, i1, i2, i3 = largest_face
    v1 = bbox_corners_local[i1] - bbox_corners_local[i0]
    v2 = bbox_corners_local[i2] - bbox_corners_local[i0]
    normal_local = v1.cross(v2).normalized()

    # Optional: if you always want the "largest face" oriented 'up' rather than 'down',
    # make sure the normal’s z-component is positive:
    if normal_local.z < 0:
        normal_local = -normal_local

    # 4) Compute the rotation that aligns this local normal to global Z=(0,0,1)
    z_axis = Vector((0, 0, 1))
    angle = normal_local.angle(z_axis)
    # if angle < 1e-6:
    #     # Already aligned or nearly so
    #     return

    rot_axis = normal_local.cross(z_axis)
    if rot_axis.length < 1e-6:
        # Edge case: normal is exactly opposite to z_axis
        rot_axis = Vector((1, 0, 0))  # or any orthogonal axis

    rot_axis.normalize()

    # Build a rotation matrix around `rot_axis` by `angle`
    rot_matrix = Matrix.Rotation(angle, 4, rot_axis)

    # Post-multiply this into the object's matrix_world
    # (i.e., rotate the object in global space).
    obj.matrix_world = obj.matrix_world @ rot_matrix

    if rand_rotate == "true":
        random_z_angle = random.uniform(0, 360)
        
        # Define the rotation angle in radians (e.g., 45 degrees)
        angle = math.radians(random_z_angle)  # Convert degrees to radians

        # Apply rotation around the global Z axis
        obj.rotation_euler[2] += angle  # Increment the Z-axis rotation

        # Update the object's transformation
        bpy.context.view_layer.update()

def apply_local_material_to_object(blend_file_path, obj):
    
    # Step 1: Load the .blend file and get the materials
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        # Import materials from the external .blend file
        data_to.materials = data_from.materials

    # Step 2: Apply the material to all objects in the given list
    if not data_to.materials:
        print("[ERROR] No materials found in the .blend file.")
        return

    # Assuming you want to apply the first material from the loaded .blend file
    material = data_to.materials[0]

    # Step 3: Iterate over the object list and apply the material to each object
    # Ensure the object is of type 'MESH'
    if obj.type == 'MESH':
        obj.data.materials.clear()
        obj.data.materials.append(material)
        print(f"[INFO] Material '{material.name}' has been applied to object '{obj.name}'.")
    else:
        print("[ERROR] obj is not mesh. Didn't add material.")


def get_object_central_point(obj):
    
    
    if obj.type != 'MESH':
        raise ValueError("The object must be of type 'MESH'.")

    
    bounding_box_world = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]

    
    center_world = sum(bounding_box_world, mathutils.Vector()) / len(bounding_box_world)

    return center_world

def load_all_objs_on_grid(obj_folder, rand_translate, rand_rotate, grid_len=2.0):
    """
    1. Clears the scene.
    2. Loads all .obj files in `obj_folder`.
    3. Places them on a 2D grid, spacing controlled by `grid_len`.
    4. Records each object's name and world location.

    Returns:
        A list of dictionaries with fields {"name": ..., "location": ...}.
    """
    clear_scene()

    if rand_translate == "true":
        # Get OBJ files and shuffle them
        obj_files = [f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")]
        num_objs = len(obj_files)
        if num_objs == 0:
            print(f"No OBJ files found in '{obj_folder}'!")
            return []

        # Generate all possible grid positions first
        grid_size = math.ceil(math.sqrt(num_objs))
        positions = [(r, c) for r in range(grid_size) for c in range(grid_size)]
        random.shuffle(positions)  # Randomize grid positions

        loaded_obj_info = []
        objs = []

        for idx, filename in enumerate(obj_files):
            if idx >= len(positions):  # Safety check
                break
                
            # Get randomized grid position
            row, col = positions[idx]
            
            # Load object
            filepath = os.path.join(obj_folder, filename)
            obj = load_obj(filepath)
            if not obj:
                continue

            # Apply material and alignment
            apply_local_material_to_object(args.blend_file_path, obj)
            align_largest_area_on_xy(obj, rand_rotate)

            # Calculate position with random grid placement
            x = col * grid_len
            y = -row * grid_len
            obj.location = (x, y, 0)

            # Record information
            loaded_obj_info.append({
                "name": filename.split('.')[0],
                "location": (x, y, 0)
            })
            objs.append(obj)

            print(f"Placed {filename} at {obj.location}")
    else:
        # 2) Gather all OBJ files
        obj_f = [f for f in os.listdir(obj_folder) if f.lower().endswith(".obj")]
        obj_files = sorted(obj_f)
        num_objs = len(obj_files)
        if num_objs == 0:
            print(f"No OBJ files found in '{obj_folder}'!")
            return []

        # 3) Compute a grid size big enough for all OBJ files
        grid_size = math.ceil(math.sqrt(num_objs))

        # This will store a dict of info for each loaded object
        loaded_obj_info = []

        objs=[]

        for idx, filename in enumerate(obj_files):
            filepath = os.path.join(obj_folder, filename)
            obj = load_obj(filepath)
            objs.append(obj)
            if not obj:
                continue  # Skip if something went wrong loading

            # (Optional) Apply a material or texture if you want:
            apply_local_material_to_object(args.blend_file_path, obj)

            # (Optional) Align largest area to XY so it "lies" flat
            align_largest_area_on_xy(obj, rand_rotate)

            # Decide row/column on the grid
            row, col = divmod(idx, grid_size)

            # x-y location on the grid
            x = col * grid_len
            y = -row * grid_len

            # # Shift the object so its bounding box sits on z=0
            # center_world = get_object_central_point(obj)
            # z = -center_world.z
            obj.location = (x, y, 0)

            # Record object info: name + location
            loaded_obj_info.append({
                "name": filename.split('.')[0],
                "location": (x, y, 0)  # or obj.location.copy() if you prefer a Vector
            })

            print(f"Placed {filename} at {obj.location}")
    
    # setup light
    light_z = 2.0
    for idx in range(num_objs):
        row, col = divmod(idx, grid_size)
        position_tuple = (col * grid_len, -row * grid_len, 0)
        offset_list = [(0.0, -grid_len / 2, light_z), (0.0, -grid_len / 2, -light_z), (grid_len / 2, 0, light_z), (grid_len / 2, 0, -light_z)]
        if row == 0:
            offset_list += [(0.0, grid_len / 2, light_z), (0.0, grid_len / 2, -light_z)]
        if col == 0:
            offset_list += [(-grid_len / 2, 0, light_z), (-grid_len / 2, 0, -light_z)]
        for offset_tuple in offset_list:
            actual_position_tuple = tuple(x+y for x, y in zip(position_tuple, offset_tuple))
            setup_environment_lighting(location=actual_position_tuple, energy=50, color=(1.0, 1.0, 1.0))

    return loaded_obj_info, objs


# with open(args.keypoints_data_path, "rb") as file:
#     keypoints = pickle.load(file)

# Load the JSON file
json_file = args.json_path
print(f"Loading JSON file: {json_file}")
with open(json_file, 'r') as f:
    data = json.load(f)

# Extract the steps from the JSON data
steps = data[int(args.iteration_index)]["steps"]
print(f"Steps extracted: {len(steps)}")

# Directory containing OBJ files
obj_root = args.input_data_dir
output_dir = args.output_data_dir
output_file = args.output_file_name
rand_translate = args.random_translate
rand_rotate = args.random_rotate
os.makedirs(output_dir, exist_ok=True)

loaded_obj_info, objs = load_all_objs_on_grid(obj_root, rand_translate, rand_rotate)
rotate_camera_around_objects(objs, loaded_obj_info)