# prompt-generator-comfyui
Custom prompt generator node for ComfyUI

# Table Of Contents
- [prompt-generator-comfyui](#prompt-generator-comfyui)
- [Table Of Contents](#table-of-contents)
- [Setup](#setup)
- [Features](#features)
- [Example Workflow](#example-workflow)
- [Variables](#variables)
  - [How Recursive Works?](#how-recursive-works)
  - [How Preprocess Mode Works?](#how-preprocess-mode-works)
    - [Example](#example)
- [Example Outputs](#example-outputs)

# Setup
- Clone the repository with ```git clone https://github.com/alpertunga-bile/prompt-generator-comfyui.git``` command under ```custom_nodes``` folder.
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` workflow
- Put your generator under ```prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Click ```Refresh``` button in ComfyUI

# Features
- Optimizations are done with [Optimum](https://github.com/huggingface/optimum) package.
- ONNX and transformers model are supported.
- Preprocessing outputs. See [this section](#how-preprocess-mode-works).
- Recursive generation is supported. See [this section](#how-recursive-works).
- Print generated text to terminal and log the node's state under  ```generated_prompts``` folder with date as filename.

# Example Workflow
![example_workflow](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/f50652a9-8751-41f3-81cf-d4cb61dd8a34)
- **Prompt Generator Node** may look different with final version but workflow is not going to change

# Variables
- You can get information about variables from [this link](https://happytransformer.com/text-generation/settings/) and [this link](https://huggingface.co/docs/transformers/v4.31.0/en/generation_strategies#text-generation-strategies).

## How Recursive Works?
- Let's say we give ```a, ``` as seed and recursive level is 1. I am going to use the same outputs for this example to understand the functionality more accurately.
- With self recursive, let's say generator's output is ```b```. So next seed is going to be ```b``` and generator's output is ```c```. Final output is ```a, c```. It can be used for generating random outputs.
- Without self recursive, let's say generator's output is ```b```. So next seed is going to be ```a, b``` and generator's output is ```c```. Final output is ```a, b, c```. It can be used for more accurate prompts.

## How Preprocess Mode Works?
- **exact_keyword** => ```(masterpiece), ((masterpiece))``` is not allowed. Checking the pure keyword without parantheses and weights. Adding prompts from the beginning of the generated text so add important prompts to seed.
- **exact_prompt** => ```(masterpiece), ((masterpiece))``` is allowed but ```(masterpiece), (masterpiece)``` is not. Checking the exact match of the prompt.
- **none** => Everything is allowed even the repeated prompts.
### Example
```
# ---------------------------------------------------------------------- Original ---------------------------------------------------------------------- #
((masterpiece)), ((masterpiece:1.2)), (masterpiece), blahblah, blah, blah, ((blahblah)), (((((blah))))), ((same prompt)), same prompt, (masterpiece)
# ------------------------------------------------------------- Preprocess (Exact Keyword) ------------------------------------------------------------- #
((masterpiece)), blahblah, blah, ((same prompt))
# ------------------------------------------------------------- Preprocess (Exact Prompt) -------------------------------------------------------------- #
((masterpiece)), ((masterpiece:1.2)), (masterpiece), blahblah, blah, ((blahblah)), (((((blah))))), ((same prompt)), same prompt
```

# Example Outputs
![ComfyUI_00062_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/82522192-b486-4703-86e2-18aff79fe29b)
![ComfyUI_00054_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/906c9c1d-d8b5-4aa7-89cc-6a1918eac454)
![ComfyUI_00048_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/e559c843-8e4c-4f45-9a39-c7f457218467)
