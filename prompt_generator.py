from os import listdir, mkdir
from os.path import join, isdir, exists

class PromptGenerator:
    def __init__(self) -> None:
        root = join("models", "prompt_generators")
        if exists(root) is False:
            print(f"{root} is created. Please add your prompt generators to {root} folder")
            mkdir(root)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model_type": ("STRING", {
                    "multiline" : False,
                    "default" : "gpt2"
                }),
                "model_name":([file for file in listdir(join("models", "prompt_generators")) if isdir(join(join("models", "prompt_generators"), file))],),
                "seed": ("STRING", {
                    "multiline" : True,
                    "default" : "((masterpiece, best quality, ultra detailed)), illustration, digital art, 1girl, solo, ((stunningly beautiful))"
                }),
                "min_length": ("INT", {
                    "default": 20,
                    "min":0,
                    "max":100,
                    "step":1
                }),
                "max_length": ("INT", {
                    "default": 50,
                    "min":35,
                    "max":200,
                    "step":1
                }),
                "do_sample": (["disable", "enable"],),
                "early_stopping": (["disable", "enable"],),
                "num_beams": ("INT", {
                    "default": 1,
                    "min":1,
                    "max":50,
                    "step":1
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "top_k": ("INT", {
                    "default": 50,
                    "min":0,
                    "max":150,
                    "step":1
                }),
                "top_p": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.1
                }),
                "no_repeat_ngram_size": ("INT", {
                    "default": 0,
                    "min":0,
                    "max":50,
                    "step":1
                }),
                "self_recursive": (["disable", "enable"],),
                "recursive_level": ("INT", {
                    "default": 0,
                    "min":0,
                    "max":50,
                    "step":1
                }),
                "preprocess_mode": (["exact_keyword", "exact_prompt", "none"],),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "generate"

    CATEGORY = "Prompt Generator"

    def RemoveDuplicates(self, line : str) -> list[str]:
        prompts = line.split(",")
        pure_prompts = []
        can_add = True

        for prompt in prompts:
            keyword = prompt.strip("(),")
            
            if keyword == "":
                continue

            for pos_prompt in pure_prompts:
                if keyword in pos_prompt:
                    can_add = False
                    break

            if can_add is False:
                continue

            pure_prompts.append(prompt)
            can_add = True

        return pure_prompts


    def Preprocess(self, line : str, preprocess_mode : str) -> str:
        from re import sub, compile
        
        pattern = compile(r'(,\s){2,}')

        temp_line = line.replace(u'\xa0', u' ')
        temp_line = temp_line.replace("\n", ", ")
        temp_line = temp_line.replace("\t", " ")
        temp_line = temp_line.replace("|", ",")
        temp_line = sub(pattern, ', ', temp_line)
        temp_line = temp_line.replace("  ", " ")

        if preprocess_mode == "exact_keyword":
            temp_line = ','.join(self.RemoveDuplicates(temp_line))
        elif preprocess_mode == "exact_prompt":
            temp_line = ','.join(list(dict.fromkeys(temp_line.split(","))))

        return temp_line
    
    def GetGeneratedText(self, generator, gen_args, seed : str, is_self_recursive : bool, recursive_level : int, preprocess_mode : str) -> str:
        result = generator.generate_text(seed, gen_args)
        generated_text = self.Preprocess(seed + result.text, preprocess_mode)

        if is_self_recursive:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text = self.Preprocess(result.text, preprocess_mode)
            generated_text = self.Preprocess(seed + generated_text, preprocess_mode)
        else:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text += result.text
                generated_text = self.Preprocess(generated_text, preprocess_mode)
            
        return generated_text
    
    def LogOutputs(self, seed : str, generated_text : str, self_recursive : str, recursive_level : int, preprocess_mode : str, gen_settings, log_filename : str) -> None:
        from datetime import datetime

        print_string = f"{'  PROMPT GENERATOR OUTPUT  '.center(200, '#')}\n{generated_text}\n{'#'*200}\n"
        print(print_string)

        log_string = f"{'#'*200}\nDate & Time : {datetime.now()}\nSeed : {seed}\nPrompt : {generated_text}\n"
        log_string += f"min_length : {gen_settings.min_length}\n"
        log_string += f"max_length : {gen_settings.max_length}\n"
        log_string += f"do_sample : {gen_settings.do_sample}\n"
        log_string += f"early_stopping : {gen_settings.early_stopping}\n"
        log_string += f"num_beams : {gen_settings.num_beams}\n"
        log_string += f"temperature : {gen_settings.temperature}\n"
        log_string += f"top_k : {gen_settings.top_k}\n"
        log_string += f"top_p : {gen_settings.top_p}\n"
        log_string += f"no_repeat_ngram_size : {gen_settings.no_repeat_ngram_size}\n"
        log_string += f"self_recursive : {self_recursive}\nrecursive_level : {recursive_level}\npreprocess_mode : {preprocess_mode}\n"
        with open(log_filename, "a") as file:
            file.write(log_string)

    def generate(self, clip, model_type, model_name, seed, min_length, max_length, do_sample, early_stopping, num_beams, temperature, top_k, top_p, no_repeat_ngram_size, self_recursive, recursive_level, preprocess_mode):
        from happytransformer import HappyGeneration, GENSettings

        root = join("models", "prompt_generators")
        real_path = join(root, model_name)
        prompt_log_filename = "generated_prompts.txt"
        generated_text = ""

        if exists(prompt_log_filename) is False:
            file = open(prompt_log_filename, "w")
            file.close()

        if exists(real_path) is False:
            print(f"{real_path} is not exists")
            generated_text = seed
        else:
            is_self_recursive = True if self_recursive == "enable" else False

            upper_model_type = model_type.upper()
            if model_type.find("/") != -1:
                upper_model_type = model_type.split("/")[1].upper()

            generator = HappyGeneration(model_type=upper_model_type, model_name=model_type, load_path=real_path)

            gen_settings = GENSettings(
                min_length=min_length,
                max_length=max_length,
                do_sample=True if do_sample == "enable" else False,
                early_stopping=True if early_stopping == "enable" else False,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

            generated_text = self.GetGeneratedText(generator, gen_settings, seed, is_self_recursive, recursive_level, preprocess_mode)

        self.LogOutputs(seed, generated_text, self_recursive, recursive_level, preprocess_mode, gen_settings, prompt_log_filename)

        tokens = clip.tokenize(generated_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

NODE_CLASS_MAPPINGS = {
    "Prompt Generator": PromptGenerator
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Prompt Generator": "Prompt Generator"
}