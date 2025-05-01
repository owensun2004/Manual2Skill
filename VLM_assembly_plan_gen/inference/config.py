import os
from pathlib import Path

# Get project root directory (assuming config.py is in the project root)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Define paths relative to project root
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SCENE_DIR = os.path.join(DATA_DIR, "preassembly_scenes")
MANUAL_DATA_PATH = os.path.join(DATA_DIR, "main_data.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Default model settings
DEFAULT_MODEL = "gpt-4o"
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0