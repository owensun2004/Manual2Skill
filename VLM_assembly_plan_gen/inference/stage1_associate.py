from openai import OpenAI
import os
import json
from datetime import datetime
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, SCENE_DIR, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from utils import encode_image, load_prompt, ensure_dir, alphanumeric_sort_key, save_json

def generate_json(img_path, manual_path, output_path, model_type=DEFAULT_MODEL, temperature=0, debug=False):
    """Generate JSON label data from images using the specified model."""
    base64_image = encode_image(img_path)
    base64_image2 = encode_image(manual_path)
    
    prompt = load_prompt("generate_json")
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_type,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image2}",
                            "detail": "high"
                        }
                    },
                ],
            }
        ],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=temperature,
    )
    final_output = response.choices[0].message.content.replace("```json", "").replace("```", "")
    
    # Save the output
    if debug:
        with open(os.path.join(output_path, "stage1_prompt_partial.txt"), "w") as f:
            f.write(prompt)
        print(f"Saved stage1 input prompt A to {os.path.join(output_path, 'stage1_prompt_partial.txt')}")
        with open(os.path.join(output_path, "stage1_output_partial.json"), "w") as f:
            f.write(final_output)
        print(f"Saved stage1 output A to {os.path.join(output_path, 'stage1_output_partial.json')}")
    
    return final_output

def select_materials_for_planning(furniture_name, furniture_type, pdf_path, 
                                  args):
    """Select materials for planning based on furniture specifications."""
    if args.scene_type == "original":
        scene_type = "scene_annotated.png"
    else:
        scene_type = "scene_rot_annotated.png"
    scene_path = os.path.join(SCENE_DIR, furniture_type, furniture_name, scene_type)

    manual_path = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name, "page_1.png")
    
    output_folder = os.path.join(pdf_path, furniture_type, furniture_name, "debug")
    ensure_dir(output_folder)
    
    
    raw_table = generate_json(scene_path, manual_path, output_folder, args.model, args.temp, args.debug)
    
    prompt_text = "select_material"
    
    # Get manual pages
    b64_pages = []
    manual_dir = os.path.join(DATA_DIR, "pdfs", furniture_type, furniture_name)
    for page in os.listdir(manual_dir):
        if page.endswith(".png"):
            page_path = os.path.join(manual_dir, page)
            b64_pages.append(encode_image(page_path))
    
    b64_pages_sorted = sorted(b64_pages, key=alphanumeric_sort_key)
    base64_image = encode_image(scene_path)
    
    prompt = load_prompt(prompt_text)
    full_prompt = prompt + "\n\nAnd here is the json file: \n" + raw_table
    
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
    
    # Add each page image to the message
    for b64_image in b64_pages_sorted:
        image_messages.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}",
                "detail": "high"
            },
        })
    
    client = OpenAI()
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    *image_messages
                ],
            }
        ],
        max_tokens=DEFAULT_MAX_TOKENS,
        temperature=args.temp
    )

    if args.debug:
        with open(os.path.join(output_folder, "stage1_prompt.txt"), "w") as f:
            f.write(full_prompt)
        print(f"Saved stage1 input prompt B to {os.path.join(output_folder, 'stage1_prompt.txt')}")
        with open(os.path.join(output_folder, "stage1_output.json"), "w") as f:
            f.write(response.choices[0].message.content.replace("```json", "").replace("```", ""))
        print(f"Saved stage1 output B to {os.path.join(output_folder, 'stage1_output.json')}")
    
    return response.choices[0].message.content