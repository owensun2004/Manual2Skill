import argparse
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from config import MANUAL_DATA_PATH, OUTPUT_DIR
from utils import load_json, encode_image
from stage1_associate import select_materials_for_planning
from stage2_planning import create_plan
from convert import convert_to_tree

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end",   type=int, default=102)
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--temp",  type=float, default=0.0)
    p.add_argument("--debug",  type=bool, default=False)
    # numbered or not_numbered
    p.add_argument("--prompt_type",  type=str, default="numbered")
    # original or varied
    p.add_argument("--scene_type",  type=str, default="original")
    return p.parse_args()

def main():
    args = parse_args()
    data = load_json(MANUAL_DATA_PATH)

    output_name = f"{datetime.now().strftime('%Y_%m_%d_%H%M%S')}"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    for idx in tqdm(range(args.start, min(args.end, len(data))), desc="Generating Assembly Graphs"):
        item = data[idx]
        name, cat = item["name"], item["category"]
        
        stage1_output = select_materials_for_planning(name, cat, output_path, args)
        stage2_output = create_plan(name, cat, output_path, stage1_output, args)

        convert_to_tree(name, cat, output_path, stage2_output, args)

        print(f"Generated Assembly Graph for Furniture Item {cat}\{name} at {output_path}/{cat}/{name}/tree.json")

if __name__ == "__main__":
    main()