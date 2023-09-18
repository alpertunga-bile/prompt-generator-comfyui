from sys import path
from os.path import dirname, exists, join
from os import mkdir
from platform import system
from folder_paths import models_dir

os_name = system()

path.append(dirname(__file__))

from prompt_generator import PromptGenerator

print("/_\ Loading Prompt Generator")

# Check prompt_generators folder under models folder

root = join(models_dir, "prompt_generators")
if exists(root) is False:
    print(f"/_\ {root} is created. Please add your prompt generators to {root} folder")
    mkdir(root)

if exists("generated_prompts") is False:
    mkdir("generated_prompts")

# Import PromptGenerator node to ComfyUI

NODE_CLASS_MAPPINGS = {
    "Prompt Generator": PromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Generator": "Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
