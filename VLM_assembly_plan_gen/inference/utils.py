import json
import re
import base64
import os

def alphanumeric_sort_key(filename):
    """Sort filenames with mixed alphanumeric content naturally."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', filename)]

def load_json(json_path):
    """Load and parse a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def save_json(data, output_path):
    """Save data to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

def encode_image(image_path):
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(directory, exist_ok=True)
    return directory

def load_prompt(prompt_name):
    """Load a prompt template from the prompts directory."""
    from config import PROMPTS_DIR
    prompt_path = os.path.join(PROMPTS_DIR, f"{prompt_name}.txt")
    with open(prompt_path, 'r') as f:
        return f.read()