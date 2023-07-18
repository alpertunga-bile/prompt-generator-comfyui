# prompt-generator-comfyui
Custom prompt generator node for ComfyUI

# Setup
- Run ```pip install happytransformer``` command in the environment that you are launching ComfyUI with
- Copy ```prompt_generator.py``` file to ```custom_nodes``` folder in ComfyUI
- Create ```prompt_generators``` folder under ```models``` folder in ComfyUI
- Put your generator under ```prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` workflow

# Features
- Print generated text to terminal and log the node's state in ```generated_prompts.txt``` file

# Example Workflow
![example_workflow](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/f50652a9-8751-41f3-81cf-d4cb61dd8a34)
- **Prompt Generator Node** may look different with final version but workflow is not going to change

# Variables
- For ```model_type``` variable copy and paste the model's name from [this site](https://huggingface.co/models?pipeline_tag=text-generation) like this ```tiiuae/falcon-40b```
- You can get information about variables from [this](https://happytransformer.com/text-generation/settings/) link

## How Recursive Works?
- Let's say we give ```a, ``` as seed and recursive level is 1. I am going to use the same outputs for this example to understand the functionality more accurately.
- With self recursive, let's say generator's output is ```b```. So next seed is going to be ```b``` and generator's output is ```c```. Final output is ```a, c```. It can be used for generating random outputs.
- Without self recursive, let's say generator's output is ```b```. So next seed is going to be ```a, b``` and generator's output is ```c```. Final output is ```a, b, c```. It can be used for more accurate prompts.

## How Preprocess Mode Works?
- **exact_keyword** => ```(masterpiece), ((masterpiece))``` is not allowed. Checking the pure keyword without parantheses and weights. Adding prompts from the beginning of the generated text so add important prompts to seed.
- **exact_prompt** => ```(masterpiece), ((masterpiece))``` is allowed but ```(masterpiece), (masterpiece)``` is not. Checking the exact match of the prompt.
- **none** => Everything is allowed even the repeated prompts.
