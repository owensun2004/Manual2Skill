from openai import OpenAI
import os
from datetime import datetime

from config import DATA_DIR, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, SCENE_DIR
from utils import encode_image, load_prompt, ensure_dir, alphanumeric_sort_key

def create_plan(furniture_name, furniture_type, pdf_path, output_table, args):
    """Create a furniture assembly plan based on manual images."""
    if args.prompt_type == "numbered":
        prompt_text = "planning_no_seg"
    else:
        prompt_text = "planning_no_seg_no_num"
    
    # Setup paths
    if args.scene_type == "original":
        scene_type = "scene_annotated.png"
    else:
        scene_type = "scene_rot_annotated.png"
    image_path = os.path.join(SCENE_DIR, furniture_type, furniture_name, scene_type)
    base64_image = encode_image(image_path)
    
    # Initialize OpenAI client
    client = OpenAI()
    
    # Load prompt text
    prompt = load_prompt(prompt_text)
    prompt = prompt + output_table
    
    # Encode manual images
    obj_img = []
    mask_dir = os.path.join(DATA_DIR, "mask", furniture_type, furniture_name)
    file_list = os.listdir(mask_dir)
    sorted_file_list = sorted(file_list, key=alphanumeric_sort_key)

    for filename in sorted_file_list:
        if args.prompt_type == "numbered" and filename.endswith("seg_numbered.png"):
            img_path = os.path.join(mask_dir, filename)
            obj_img.append(encode_image(img_path))
        elif args.prompt_type == "not_numbered" and filename.endswith("seg.png"):
            img_path = os.path.join(mask_dir, filename)
            obj_img.append(encode_image(img_path))

    # Build image messages
    image_messages = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            },
        }
    ]
        
    for b64_obj_image in obj_img:
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_obj_image}",
                "detail": "high"
            },
        })
    
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_messages
                ],
            }
        ],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=args.temp,
    )
    
    # Save response
    if args.debug:
        output_folder = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
        ensure_dir(output_folder)
        
        with open(os.path.join(output_folder, "stage2_prompt.txt"), "w") as f:
            f.write(prompt)
        print(f"Saved stage2 input prompt to {os.path.join(output_folder, 'stage2_prompt.txt')}")
        with open(os.path.join(output_folder, "stage2_output.txt"), "w") as f:
            f.write(response.choices[0].message.content)
        print(f"Saved stage2 output to {os.path.join(output_folder, 'stage2_output.txt')}")
    
    return response.choices[0].message.content