# prompt-generator-comfyui
Custom prompt generator node for ComfyUI

# Setup
- Run ```pip install happytransformer``` command in the environment that you are launching ComfyUI with
- Copy ```prompt_generator.py``` file to ```custom_nodes``` folder in ComfyUI
- Create ```prompt_generators``` folder under ```models``` folder in ComfyUI
- Put your generator under ```prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser)
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` workflow

# Features
- Print generated text to terminal and log the node's state in ```generated_prompt.txt``` file

# Example Workflow
![example_workflow](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/f50652a9-8751-41f3-81cf-d4cb61dd8a34)

# Variables
- You can get info from variables from [this](https://happytransformer.com/text-generation/) and [this](https://happytransformer.com/text-generation/settings/) links

## How Recursive Works?
- Let's say we give ```a, ``` as seed and recursive level is 1. I am going to use the same outputs for this example to understand the functionality more accurately.
- With self recursive, let's say generator's output is ```b```. So next seed is going to be ```b``` and generator's output is ```c```. Final output is ```a, c```. It can be used for generating random outputs.
- Without self recursive, let's say generator's output is ```b```. So next seed is going to be ```a, b``` and generator's output is ```a, b, c```. Final output is ```a, b, c```. It can be used for more accurate prompts.
