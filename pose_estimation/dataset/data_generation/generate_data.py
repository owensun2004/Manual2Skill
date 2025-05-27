import subprocess
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
from tqdm import tqdm

def call_python_script(script_path, **kwargs):
    # Create the command line string
    command = ['python', script_path]
    
    # Add --param_name argument for each named parameter
    for key, value in kwargs.items():
        command.append(f'--{key}')
        command.append(str(value))

    # Call the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Output the result of the command execution
    if result.returncode == 0:
        pass
    else:
        print("Script execution failed")
        print(result.stderr)  # Print standard error


def call_blender_script(script_path, **kwargs):
    # Create the command line string
    command = ['/data2/lyw/blender-4.3.2-linux-x64/blender', '-b', '-P', script_path]
    
    # Add --param_name argument for each named parameter
    for key, value in kwargs.items():
        command.append(str(value))

    # Call the command
    result = subprocess.run(command, capture_output=True, text=True)
    
    # Output the result of the command execution
    if result.returncode == 0:
        pass


if __name__ == "__main__":

    #! README: This script is used to generate data for training the model.
    #! README: The data generation process is divided into three steps:
    #! README: 1. Generate the graph data of the parts
    #! README: 2. Render the parts with random colors and camera poses
    #! README: 3. Process the rendered images and generate the training data
    #! README: input_mesh_dir: The directory containing the original mesh files of the parts
    #! README: dataset_dir: The directory to save the generated dataset
    #! README: num_parts_selection: The number of parts combinations to generate (may not be the final number of combinations due to selection algorithm)
    #! README: image_per_selection: The number of images to render for each part combination
    #! README: num_data_per_image: The number of data to generate for each image
    #! README: The total number of data pieces generated will be num_parts_selection * image_per_selection * num_data_per_image
    
    # partnet/partnet_dataset/stats/new_raw_data/ is the original dataset directory
    input_dir = '/data2/lyw/rss_data/partnet/partnet_dataset/stats/new_raw_data/' 

    # /data2/lyw/partnet_mono is the target saved directory 
    dataset_dir = os.path.abspath('/data2/lyw/partnet_mono')
    os.makedirs(dataset_dir, exist_ok=True)

    # Target classes and their quantities
    target_classes = {"Chair": 100, "Lamp": 100, "Table": 100}
    selected_folders = {key: [] for key in target_classes}  # Store the selected folders

    # Traverse the directories
    all_folders = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    # Filter folders by target classes
    for folder in all_folders:
        folder_path = os.path.join(input_dir, folder)
        
        # Get the number of files
        num_files = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        # Add to selected folders only if it matches the category name and the number of files <= 5
        for category in target_classes:
            if category in folder and num_files <= 5:
                if len(selected_folders[category]) < target_classes[category]:
                    selected_folders[category].append(folder)

    selected_folder_list = []
    for category, folders in selected_folders.items():
        selected_folder_list.extend(folders)

    num_parts_selection = 10
    image_per_selection = 20
    image_res_H = 576
    image_res_W = 576
    radius = 2.0

    script1_path = os.path.abspath('./graph_generation.py')  # Path to the target Python script
    script2_path = os.path.abspath('./render.py')  # Path to the target Python script
    script3_path = os.path.abspath('./process_data.py')  # Path to the target Python script
    
    for folder in tqdm(selected_folder_list):
        input_mesh_dir = os.path.join(input_dir, folder)
        call_python_script(script1_path, input_data_dir=input_mesh_dir, 
                        num_parts_selection=num_parts_selection,
                        save_data_dir = os.path.join(dataset_dir, folder))
        call_blender_script(script2_path, input_data_dir=os.path.join(dataset_dir, folder),
                        image_per_selection=image_per_selection, image_res_H_=image_res_H, image_res_W_=image_res_W,
                        radius=radius)
        part_selections = os.listdir(os.path.join(dataset_dir, folder))
        for part_selection in part_selections:
            call_python_script(script3_path, data_dir=os.path.join(os.path.join(dataset_dir, folder), part_selection))
