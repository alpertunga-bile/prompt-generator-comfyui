from sys import path
from os.path import dirname, exists, join
from os import mkdir
from platform import system
from folder_paths import models_dir, base_path

os_name = system()

path.append(dirname(__file__))

from prompt_generator import PromptGenerator

print("/_\ Loading Prompt Generator")

# check prompt_generators folder under the models folder

root = join(models_dir, "prompt_generators")
if exists(root) is False:
    mkdir(root)
    print(f"/_\ {root} is created. Please add your prompt generators to {root} folder")

prompts_file = join(base_path, "generated_prompts")
if exists(prompts_file) is False:
    mkdir(prompts_file)

# import PromptGenerator node to ComfyUI

NODE_CLASS_MAPPINGS = {
    "Prompt Generator": PromptGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Generator": "Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("/_\ Loaded Successfully")
print("-" * 100)
