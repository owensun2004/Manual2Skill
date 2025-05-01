import json
import subprocess
import os
import argparse

# Path to the JSON file containing the directories
json_file_path = os.path.join(os.path.dirname(__file__), "../data", "main_data.json")

# Path to the scripts
script_blender = os.path.join(os.path.dirname(__file__), "renderer.py")
script_annotate = os.path.join(os.path.dirname(__file__), "anotator.py")

# Pickle path
pkl_path = os.path.join(os.path.dirname(__file__), "data.pkl")

def load_paths(json_path):
    """Load directory paths from the JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rand_translate", type=str, default="false")
    p.add_argument("--rand_rotate", type=str, default="false")
    args = p.parse_args()

    # Load all paths from the JSON file
    data = load_paths(json_file_path)

    gpu_id = "0"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_id}

    output_file = "scene_rot_annotated.png"

    for i in range(0, len(data)):
        furniture_name = data[i]["name"]
        furniture_cat = data[i]["category"]

        path = os.path.join(os.path.dirname(__file__), "data", "parts", furniture_cat, furniture_name)

        subprocess.run([
            "blender", 
            "--background",
            "--python", script_blender, 
            "--",  # Separate Blender options from script arguments
            "--random_translate", args.rand_translate,
            "--random_rotate", args.rand_rotate,
            "--input_data_dir", path, 
            "--iteration_index", str(i),
            "--json_path", json_file_path,
            "--keypoints_data_path", pkl_path,
            "--output_data_dir", os.path.join(os.path.dirname(__file__), "../data/pre-assembly_scenes", furniture_cat, furniture_name),
            "--output_file_name", output_file,
            "--blend_file_path", os.path.join(os.path.dirname(__file__), "blender_materials", "wood_2K_4ba225a0-3c19-44e8-80f1-33736bacb14f.blend"),
            "--no_3d_keypoints",
        ], check=True, env=env)
        subprocess.run([
            "conda", "run", "-n", "fa", "python",
            script_annotate,
            "--input_data_dir", os.path.join(os.path.dirname(__file__), "../data/pre-assembly_scenes", furniture_cat, furniture_name),
            "--output_file_name", output_file,
            "--no_3d_keypoints",
        ], check=True)


if __name__ == "__main__":

    main()