# prompt-generator-comfyui
Custom prompt generator node for ComfyUI

# Setup
- Run ```pip install happytransformer``` command in the environment that you are launching ComfyUI with
- Copy ```prompt_generator.py``` file to custom_nodes folder in ComfyUI
- Create ```prompt_generators``` folder under models folder in ComfyUI
- Put your generator under ```prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser)
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` workflow

# Features
- Print generated text to terminal and log the node's state in ```generated_prompt.txt``` file
