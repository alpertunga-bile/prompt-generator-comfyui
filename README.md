# prompt-generator-comfyui
Custom AI prompt generator node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI). With this node, you can use text generation models to generate prompts. Before using, text generation model has to be trained with prompt dataset.

# Table Of Contents
- [prompt-generator-comfyui](#prompt-generator-comfyui)
- [Table Of Contents](#table-of-contents)
- [Setup](#setup)
  - [For Portable Version of the ComfyUI](#for-portable-version-of-the-comfyui)
  - [For Manual Installation of the ComfyUI](#for-manual-installation-of-the-comfyui)
  - [For ComfyUI Manager Users](#for-comfyui-manager-users)
- [Features](#features)
- [Example Workflow](#example-workflow)
- [Pretrained Prompt Models](#pretrained-prompt-models)
  - [Dataset](#dataset)
  - [Models](#models)
- [Variables](#variables)
  - [How Recursive Works?](#how-recursive-works)
  - [How Preprocess Mode Works?](#how-preprocess-mode-works)
    - [Example](#example)
- [Troubleshooting](#troubleshooting)
  - [Package Version](#package-version)
    - [For Manual Installation of the ComfyUI](#for-manual-installation-of-the-comfyui-1)
    - [For Portable Installation of the ComfyUI](#for-portable-installation-of-the-comfyui)
  - [Automatic Installation](#automatic-installation)
    - [For Manual Installation of the ComfyUI](#for-manual-installation-of-the-comfyui-2)
    - [For Portable Installation of the ComfyUI](#for-portable-installation-of-the-comfyui-1)
  - [New Updates On The Node](#new-updates-on-the-node)
- [Contributing](#contributing)
- [Example Outputs](#example-outputs)

# Setup
## For Portable Version of the ComfyUI
- [x] Automatic installation is added for portable version.
- Clone the repository with ```git clone https://github.com/alpertunga-bile/prompt-generator-comfyui.git``` command under ```custom_nodes``` folder.
- Run the **run_nvidia_gpu.bat** file
- Open the ```hires.fixWithPromptGenerator.json``` or ```basicWorkflowWithPromptGenerator.json``` workflow
- Put your generator under ```models/prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Click ```Refresh``` button in ComfyUI

## For Manual Installation of the ComfyUI
- Clone the repository with ```git clone https://github.com/alpertunga-bile/prompt-generator-comfyui.git``` command under ```custom_nodes``` folder.
- Run the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` or ```basicWorkflowWithPromptGenerator.json``` workflow
- Put your generator under ```models/prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Click ```Refresh``` button in ComfyUI

## For ComfyUI Manager Users
- Download the node with [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- Restart the ComfyUI
- Open the ```hires.fixWithPromptGenerator.json``` or ```basicWorkflowWithPromptGenerator.json``` workflow
- Put your generator under ```models/prompt_generators``` folder. You can create your prompt generator with [this repository](https://github.com/alpertunga-bile/prompt-markdown-parser). You have to put generator as folder. Do not just put ```pytorch_model.bin``` file for example.
- Click ```Refresh``` button in ComfyUI

# Features
- Multiple output generation is added. You can choose from 5 outputs and check the generated prompts in the log file and terminal. The prompts are logged and printed in order.
- Optimizations are done with [Optimum](https://github.com/huggingface/optimum) package.
- ONNX and transformers models are supported.
- Preprocessing outputs. See [this section](#how-preprocess-mode-works).
- Recursive generation is supported. See [this section](#how-recursive-works).
- Print generated text to terminal and log the node's state under  ```generated_prompts``` folder with date as filename.

# Example Workflow
![example_workflow](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/f50652a9-8751-41f3-81cf-d4cb61dd8a34)

![example_workflow_basic](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/544907ff-2a75-415a-912c-90dfb6959efb)

- **Prompt Generator Node** may look different with final version but workflow in ComfyUI is not going to change

# Pretrained Prompt Models

- You can find the models in [this link](https://drive.google.com/drive/folders/1c21kMH6FTaia5C8239okL3Q0wJnnWc1N?usp=share_link)

- For to use the pretrained model follow these steps:
  - Download the model and unzip to ```models/prompt_generators``` folder.
  - Click ```Refresh``` button in ComfyUI.
  - Then select the generator with the node's ```model_name``` variable (If you can't see the generator restart the ComfyUI).

## Dataset

- 1.631.670 rows of unique prompts | **(Gathering In Process)**
- %85 train | %15 test
- Process of data cleaning and gathering can be found [here](https://github.com/alpertunga-bile/prompt-markdown-parser/blob/master/CLI/CLICivitai.py)
- These Huggingface datasets are used for enhancing the dataset:
  - [FredZhang7/anime-prompts-180K](https://huggingface.co/datasets/FredZhang7/anime-prompts-180K)
  - [FredZhang7/stable-diffusion-prompts-2.47M](https://huggingface.co/datasets/FredZhang7/stable-diffusion-prompts-2.47M)
  - [thefcraft/civitai-stable-diffusion-337k](https://huggingface.co/datasets/thefcraft/civitai-stable-diffusion-337k)
  - [Falah/stable_diffusion_prompts_dataset](https://huggingface.co/datasets/Falah/stable_diffusion_prompts_dataset)
  - [daspartho/stable-diffusion-prompts](https://huggingface.co/datasets/daspartho/stable-diffusion-prompts)
  - [andyyang/stable_diffusion_prompts_2m](https://huggingface.co/datasets/andyyang/stable_diffusion_prompts_2m)
  - [poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb)
  - [yizhangliu/stable-diffusion-prompts](https://huggingface.co/datasets/yizhangliu/stable-diffusion-prompts)
    
## Models

- female_positive_generator_v2 | **(Training In Process)**
  - using [distilgpt2](https://huggingface.co/distilgpt2) model
  - Training Loss ~0.50
  
- female_positive_generator_v3 | **(Training In Process)**
  - using [bigscience/bloom-560m](https://huggingface.co/bigscience/bloom-560m) model
  - Training loss ~0.59

# Variables

- ```num_beams``` must be dividable by ```num_beam_groups``` otherwise you will get errors.

|      Variable Names       | Definitions                                                                                                                                                                                                                                                             |
| :-----------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|      **model_name**       | Folder name that contains the model                                                                                                                                                                                                                                     |
|      **accelerate**       | Open optimizations. Some of the models are not supported by BetterTransformer ([Check your model](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models)). If it is not supported switch this option to disable or convert your model to ONNX |
|        **prompt**         | Input prompt for the generator                                                                                                                                                                                                                                          |
|          **cfg**          | CFG is enabled by setting guidance_scale > 1. Higher guidance scale encourages the model to generate samples that are more closely linked to the input prompt, usually at the expense of poorer quality                                                                 |
|    **min_new_tokens**     | The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.                                                                                                                                                                                 |
|    **max_new_tokens**     | The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.                                                                                                                                                                                 |
|       **do_sample**       | When True, picks words based on their conditional probability                                                                                                                                                                                                           |
|    **early_stopping**     | When True, generation finishes if the EOS token is reached                                                                                                                                                                                                              |
|       **num_beams**       | Number of steps for each search path                                                                                                                                                                                                                                    |
|    **num_beam_groups**    | Number of groups to divide num_beams into in order to ensure diversity among different groups of beams                                                                                                                                                                  |
|   **diversity_penalty**   | This value is subtracted from a beam’s score if it generates a token same as any beam from other group at a particular time. Note that diversity_penalty is only effective if ```group beam search``` is enabled.                                                       |
|      **temperature**      | How sensitive the algorithm is to selecting low probability options                                                                                                                                                                                                     |
|         **top_k**         | How many potential answers are considered when performing sampling                                                                                                                                                                                                      |
|         **top_p**         | Min number of tokens are selected where their probabilities add up to top_p                                                                                                                                                                                             |
|  **repetition_penalty**   | The parameter for repetition penalty. 1.0 means no penalty                                                                                                                                                                                                              |
| **no_repeat_ngram_size**  | The size of an n-gram that cannot occur more than once. (0=infinity)                                                                                                                                                                                                    |
| **remove_invalid_values** | Whether to remove possible nan and inf outputs of the model to prevent the generation method to crash. Note that using remove_invalid_values can slow down generation.                                                                                                  |
|    **self_recursive**     | See [this section](#how-recursive-works)                                                                                                                                                                                                                                |
|    **recursive_level**    | See [this section](#how-recursive-works)                                                                                                                                                                                                                                |
|    **preprocess_mode**    | See [this section](#how-preprocess-mode-works)                                                                                                                                                                                                                          |

- For more information, follow [this link](https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/text_generation#transformers.GenerationConfig)
- Look [this link](https://huggingface.co/docs/transformers/v4.31.0/en/generation_strategies#text-generation-strategies) for text generation strategies

## How Recursive Works?
- Let's say we give ```a, ``` as seed and recursive level is 1. I am going to use the same outputs for this example to explain the functionality more accurately.
- With self recursive, let's say generator's output is ```b```. So next seed is going to be ```b``` and generator's output is ```c```. Final output is ```a, c```. It can be used for generating random outputs.
- Without self recursive, let's say generator's output is ```b```. So next seed is going to be ```a, b``` and generator's output is ```c```. Final output is ```a, b, c```. It can be used for more accurate prompts.

## How Preprocess Mode Works?
- **exact_keyword** => ```(masterpiece), ((masterpiece))``` is not allowed. Checking the pure keyword without parantheses and weights. The algorithm is adding the prompts from the beginning of the generated text, so add important prompts to seed.
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

# Troubleshooting

- If the below solutions are not fixed your issue please create an issue with ```bug``` label

## Package Version

- The node is based on [transformers](https://github.com/huggingface/transformers) and [optimum](https://github.com/huggingface/optimum) packages. So most of the problems may be caused from these packages. For overcome these problems you can try to update these packages:

### For Manual Installation of the ComfyUI

1. Activate the virtual environment if there is one.
2. Run ```pip install --upgrade transformers optimum optimum[onnxruntime-gpu]``` command.

### For Portable Installation of the ComfyUI

1. Go to the ```ComfyUI_windows_portable``` folder.
2. Open the command prompt in this folder.
3. Run ```.\python_embeded\python.exe -s -m pip install --upgrade transformers optimum optimum[onnxruntime-gpu]``` command.

## Automatic Installation

### For Manual Installation of the ComfyUI

- The users have to check if they activate the virtual environment if there is one

### For Portable Installation of the ComfyUI

- The users have to check that they are starting the ComfyUI in the ```ComfyUI_windows_portable```
- Because the node is checking the ```python_embeded``` folder if it is exists and is using it to install the required packages

## New Updates On The Node

- Sometimes the variables are changed with updates, so it may broke the workflow. But don't worry you have to just delete the node in the workflow and add it again.

# Contributing

- Contributions are welcome. If you have an idea and want to implement it by yourself please follow these steps:

  1. Create a fork
  2. Create a branch with a name that describes the feature that you are adding
  3. Pull request the fork with the comment that explaining the new feature

- If you have an idea but don't know how to implement it please create an issue with ```enhancement``` label.

- [x] The contributing can be done in several ways. You can contribute to code or to README file.

# Example Outputs
![ComfyUI_00062_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/82522192-b486-4703-86e2-18aff79fe29b)
![ComfyUI_00055_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/af1be3ad-306d-41a1-9022-6401d394b1df)
![ComfyUI_00054_](https://github.com/alpertunga-bile/prompt-generator-comfyui/assets/76731692/906c9c1d-d8b5-4aa7-89cc-6a1918eac454)

