import bpy
import os
import random
import bpy
import numpy as np
from mathutils import Color, Vector
import json
import argparse
import sys
import os
from pdb import set_trace as bp
colors_hex = [
        '#5A9BD5', '#FF6F61', '#77B77A', '#A67EB1', '#FF89B6', '#FFB07B',
        '#C5A3CF', '#FFA8B6', '#A3C9E0', '#FFC89B', '#E58B8B',
        '#A3B8D3', '#D4C3E8', '#66B2AA', '#E4A878', '#6882A4', '#D1AEDD', '#E8A4A6',
        '#A5DAD7', '#C6424A', '#E1D1F4', '#FFD8DC', '#F4D49B', '#8394A8'
    ]


def find_partition(node, partitions):
    for i, partition in enumerate(partitions):
        if node in partition:
            return i
    return -1 


def hex_to_rgb(hex_color):
    """
    Convert hexadecimal color code to RGB tuple.
    :param hex_color: Hexadecimal color code (e.g. '#FFFFFF')
    :return: RGB color tuple with values from 0-255
    """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to randomly place the camera on the sphere and render an image
def capture_image(output_dir, radius):
    import bpy
    import os
    import numpy as np
    import random
    from mathutils import Matrix, Vector

    # Create a new orthographic camera
    camera_data = bpy.data.cameras.new("OrthographicCamera")
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 4  # Adjust for desired zoom level

    # Create camera object
    camera_object = bpy.data.objects.new("Camera", camera_data)

    # Define camera position in spherical coordinates
    possible_theta = [np.pi / 4, np.pi / 4 * 3, np.pi / 4 * 5, np.pi / 4 * 7]
    theta = random.choice(possible_theta) + random.uniform(-np.pi / 12, np.pi / 12)
    phi = random.uniform(np.pi / 6, np.pi / 3)
 

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    camera_object.location = (x, y, z)
    camera_object.data.ortho_scale = radius

    # Make the camera look at the origin
    direction = camera_object.location
    camera_object.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()

    # Add random rotation along the Z axis
    possible_z_rotations = [0, np.pi / 2, np.pi, np.pi * 3 / 2]
    random_z_rotation = random.choice(possible_z_rotations)
    camera_object.rotation_euler.rotate_axis("Z", random_z_rotation)

    # Link the camera to the scene
    bpy.context.scene.collection.objects.link(camera_object)

    # Force Blender to update the scene
    bpy.context.view_layer.update()

    # Extract the updated 4x4 transformation matrix
    updated_matrix_world = camera_object.matrix_world

    # Save the updated 4x4 matrix
    np.savetxt(os.path.join(output_dir, "camera_pose.txt"), np.array(updated_matrix_world))

    # Set the active camera
    bpy.context.scene.camera = camera_object

    # Set output file path and render the scene
    bpy.context.scene.render.filepath = os.path.join(output_dir, "image.png")
    bpy.ops.render.render(write_still=True)

    # Remove the camera after rendering
    bpy.data.objects.remove(camera_object)

print(sys.argv)

if len(sys.argv) > 1:
    input_data_dir = sys.argv[4] 
    num_images_per_selection = int(sys.argv[5]) 
    img_res_H = int(sys.argv[6])
    img_res_W = int(sys.argv[7])
    radius = float(sys.argv[8])
    # print(input_data_dir)
    # print(num_images_per_selection)
    # print(img_res_H)
    # print(img_res_W)
    # print(radius)


input_data_dir_list = [os.path.join(input_data_dir, d) for d in os.listdir(input_data_dir) if os.path.isdir(os.path.join(input_data_dir, d))]
# print(input_data_dir_list)

for data_dir in input_data_dir_list:
    # Clear all objects in the current scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    ## Set the folder path
    #folder_path = "/home/crtie/crtie/Furniture-Assembly/dataset/parts/Chair/poang_2"  # Replace with your folder path

    ## List all .obj files in the folder
    #obj_files = [os.path.abspath(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith(".obj")]

    with open(data_dir+'/part_selection.json', "r") as f:
        loaded_data = json.load(f)

    # Obtain the list of "used meshes"
    obj_files = loaded_data["used meshes"]
    groups = loaded_data["groups"]
    # print(obj_files, groups)
    # Import .obj files and assign random colors
    for obj_file in obj_files:
        # Construct the full path to the .obj file
        obj_path = os.path.join(data_dir, obj_file)
        obj_idx = int(obj_file[-5:-4])
        group_id = find_partition(obj_idx, groups)
        _color = hex_to_rgb(colors_hex[group_id % len(colors_hex)])

        # Import the .obj file
        bpy.ops.wm.obj_import(filepath=obj_path)

        # Get the imported object (assuming it's the last object imported)
        imported_object = bpy.context.selected_objects[0]
        
        # Clear any existing materials
        imported_object.data.materials.clear()

        # Create a new material
        material = bpy.data.materials.new(name=f"Material_{obj_file}")

        # Set the material to use nodes
        material.use_nodes = True

        # # Set a random color for the material
        # color = Color((_color[0] / 255, _color[1] / 255, _color[2] / 255))
        
        # # Set the color of the material
        # bsdf = material.node_tree.nodes["Principled BSDF"]
        # bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1)

        # Assign the material to the object
        if imported_object.data.materials:
            imported_object.data.materials[0] = material
        else:
            imported_object.data.materials.append(material)

    # Set World Lighting (Uniform Ambient Light)
    world = bpy.context.scene.world

    # Enable the use of the world background
    world.use_nodes = True

    # Set a light background color (the color will give a uniform light)
    background_node = world.node_tree.nodes.get("Background")
    if background_node:
        # Set the background to a light color (white or a soft light color)
        background_node.inputs["Color"].default_value = (1, 1, 1, 1)  # White light
        # Increase the strength to ensure uniform lighting
        background_node.inputs["Strength"].default_value = 1  # Increase the strength for more brightness

    # Set the render engine to Eevee or Cycles
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Or 'CYCLES'
    # Enable Freestyle rendering
    scene = bpy.context.scene
    scene.render.use_freestyle = True  # Enable freestyle
    scene.render.resolution_x = int(img_res_H)
    scene.render.resolution_y = int(img_res_W)

    # # Customize Freestyle settings
    # freestyle_settings = scene.freestyle_settings

    # # You can adjust these options to modify how the freestyle lines are drawn
    # freestyle_settings.linesets.new(name="LineSet")

    # line_set = freestyle_settings.linesets[-1]  # Get the newly created lineset

    # # Enable or disable specific line types
    # line_set.select_by_name = "Only Selected"  # Example: Only selected objects will have freestyle lines

    # # Set a simple line style for freestyle rendering
    # line_style = line_set.linestyle
    # line_style.thickness = 2  # Adjust thickness of the freestyle lines
    # line_style.color = (0, 0, 0, 1)  # Set line color to black


    # Optionally, you can set the scene's ambient lighting directly
    bpy.context.scene.world.color = (1, 1, 1)  # Set the world background color to white

    # Set the view transform to 'Standard' in Color Management
    bpy.context.scene.view_settings.view_transform = 'Standard'

    # Render settings
    bpy.context.scene.render.image_settings.file_format = 'PNG'  # Save images as PNG



    # List all the subdirectories in the parent directory
    subdirectories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Filter and get only the directories that are named with numbers
    numbered_dirs = []

    for dir_name in subdirectories:
        try:
            # Try to convert directory name to an integer
            dir_number = int(dir_name)
            numbered_dirs.append(dir_number)
        except ValueError:
            # Skip any directories that do not have integer names
            continue

    # Find the largest number (if there are any numbered directories)
    if numbered_dirs:
        max_number = max(numbered_dirs)
    else:
        max_number = -1  # If there are no numbered folders, start from -1

    # Create a new folder with the next number
    new_folder_name = str(max_number + 1)
    max_number = max_number + 1
    new_folder_path = os.path.join(data_dir, new_folder_name)

    # Create the new folder
    os.makedirs(new_folder_path)
    # Capture 10 images with random camera positions
    for i in range(num_images_per_selection):
        capture_image(new_folder_path, radius)
        if i != num_images_per_selection - 1:
            # Create a new folder with the next number
            new_folder_name = str(max_number + 1)
            new_folder_path = os.path.join(data_dir, new_folder_name)
            max_number = max_number + 1
            os.makedirs(new_folder_path, exist_ok=True)
