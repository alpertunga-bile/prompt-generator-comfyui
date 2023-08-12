from sys import path
from os.path import dirname, exists, join
from subprocess import run
from os import mkdir
from importlib.util import find_spec
from platform import system

os_name = system()

path.append(dirname(__file__))

from prompt_generator import PromptGenerator
from aspect_node import AspectNode


def check_package(package_name: str) -> None:
    if find_spec(package_name) is None:
        print(f"/_\ Installing {package_name}")
        process = run(
            f"pip install {package_name}", shell=True, check=True, capture_output=True
        )


print("/_\ Loading Prompt Generator")

# Check prompt_generators folder under models folder

root = join("models", "prompt_generators")
if exists(root) is False:
    print(f"/_\ {root} is created. Please add your prompt generators to {root} folder")
    mkdir(root)

if exists("generated_prompts") is False:
    mkdir("generated_prompts")

# Check required packages

# Installing from git because there is a error with normal transformers package
if find_spec("transformers") is None:
    print(f"/_\ Installing transformers")
    process = run(
        f"pip install git+https://github.com/huggingface/transformers",
        shell=True,
        check=True,
        capture_output=True,
    )

check_package("accelerate")
check_package("xformers")

if os_name == "Linux":
    check_package("triton")

check_package("optimum")

if find_spec("onnxruntime") is None:
    print(f"/_\ Installing onnxruntime")
    process = run(
        f"pip install optimum[onnxruntime]", shell=True, check=True, capture_output=True
    )

if find_spec("onnxruntime-gpu") is None:
    print(f"/_\ Installing onnxruntime-gpu")
    process = run(
        f"pip install optimum[onnxruntime-gpu]",
        shell=True,
        check=True,
        capture_output=True,
    )

# Import PromptGenerator node to ComfyUI

NODE_CLASS_MAPPINGS = {"Prompt Generator": PromptGenerator, "Aspect": AspectNode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Generator": "Prompt Generator",
    "Aspect": "Aspect Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
