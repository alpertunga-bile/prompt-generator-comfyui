from sys import path
from os.path import dirname, exists, join
from subprocess import run
from os import remove, mkdir

path.append(dirname(__file__))

from prompt_generator import PromptGenerator

print("/_\ Loading Prompt Generator")

# Check prompt_generators folder under models folder

root = join("models", "prompt_generators")
if exists(root) is False:
    print(f"/_\ {root} is created. Please add your prompt generators to {root} folder")
    mkdir(root)

if exists("generated_prompts") is False:
    mkdir("generated_prompts")

# Check happytranformer package

temp_requirements_file = "temp_requirements.txt"

process = run(f"pip freeze > {temp_requirements_file}", shell=True, check=True, capture_output=True)
need_to_install = True
packages = set()

with open(temp_requirements_file, "r") as file:
    packages = set(file.readlines())

for package in packages:
    if "happytransformer" in package:
        need_to_install = False
        break

remove(temp_requirements_file)

if need_to_install:
    print("/_\ Installing happytransformer")
    process = run("pip install happytransformer", shell=True, check=True, capture_output=True)

# Import PromptGenerator node to ComfyUI

NODE_CLASS_MAPPINGS = {
    "Prompt Generator": PromptGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Generator": "Prompt Generator"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']