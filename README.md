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
## For Portable Version of the ComfyUI
- [ ] Portable version users use these commands for now. I am going to add automation for this in the new commit.
- Open cmd in the **ComfyUI_windows_portable** folder.
- Execute these commands step by step:
```
.\python_embeded\python.exe -s -m pip install transformers
.\python_embeded\python.exe -s -m pip install accelerate
.\python_embeded\python.exe -s -m pip install optimum
.\python_embeded\python.exe -s -m pip install optimum[onnxruntime-gpu]
```
- Run the **run_nvidia_gpu.bat** file

## For Manual Installation of the ComfyUI
- Clone the repository with ```git clone https://github.com/alpertunga-bile/prompt-generator-comfyui.git``` command under ```custom_nodes``` folder.
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` workflow
- Put your generator under ```prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Click ```Refresh``` button in ComfyUI

# Features
- Optimizations are done with [Optimum](https://github.com/huggingface/optimum) package.
- ONNX and transformers models are supported.
- Preprocessing outputs. See [this section](#how-preprocess-mode-works).
- Recursive generation is supported. See [this section](#how-recursive-works).
- Print generated text to terminal and log the node's state under  ```generated_prompts``` folder with date as filename.

# Example Workflow
![example_workflow](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/f50652a9-8751-41f3-81cf-d4cb61dd8a34)
- **Prompt Generator Node** may look different with final version but workflow in ComfyUI is not going to change

# Variables

|      Variable Names       | Definitions                                                                                                                                                                                                                                                             |
| :-----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      **model_name**       | Folder name that contains the model                                                                                                                                                                                                                                     |
|      **accelerate**       | Open optimizations. Some of the models are not supported by BetterTransformer ([Check your model](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models)). If it is not supported switch this option to disable or convert your model to ONNX |
|        **prompt**         | Input prompt for the generator                                                                                                                                                                                                                                          |
|          **cfg**          | CFG is enabled by setting guidance_scale > 1. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality                                                                 |
|      **min_length**       | Minimum number of generated tokens                                                                                                                                                                                                                                      |
|      **max_length**       | Maximum number of generated tokens                                                                                                                                                                                                                                      |
|       **do_sample**       | When True, picks words based on their conditional probability                                                                                                                                                                                                           |
|    **early_stopping**     | When True, generation finishes if the EOS token is reached                                                                                                                                                                                                              |
|       **num_beams**       | Number of steps for each search path                                                                                                                                                                                                                                    |
|    **num_beam_groups**    | Number of groups to divide num_beams into in order to ensure diversity among different groups of beams                                                                                                                                                                  |
|      **temperature**      | How sensitive the algorithm is to selecting low probability options                                                                                                                                                                                                     |
|         **top_k**         | How many potential answers are considered when performing sampling                                                                                                                                                                                                      |
|         **top_p**         | Min number of tokens are selected where their probabilities add up to top_p                                                                                                                                                                                             |
|  **repetition_penalty**   | The parameter for repetition penalty. 1.0 means no penalty                                                                                                                                                                                                              |
| **no_repeat_ngram_size**  | The size of an n-gram that cannot occur more than once. (0=infinity)                                                                                                                                                                                                    |
| **remove_invalid_values** | Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.                                                                                                  |
|    **self_recursive**     | See [this section](#how-recursive-works)                                                                                                                                                                                                                                |
|    **recursive_level**    | See [this section](#how-recursive-works)                                                                                                                                                                                                                                |
|    **preprocess_mode**    | See [this section](#how-preprocess-mode-works)                                                                                                                                                                                                                          |

- For more information, look [this link](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig)

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
