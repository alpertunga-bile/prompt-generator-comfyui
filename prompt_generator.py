class PromptGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model_type": ("STRING", {
                    "multiline" : False,
                    "default" : "gpt2"
                }),
                "model_name": ("STRING", {
                    "multiline" : False,
                }),
                "seed": ("STRING", {
                    "multiline" : True,
                    "default" : "mature woman"
                }),
                "min_token": ("INT", {
                    "default": 5,
                    "min":0,
                    "max":20,
                    "step":1
                }),
                "max_token": ("INT", {
                    "default": 30,
                    "min":20,
                    "max":50,
                    "step":1
                }),
                "do_sample": (["enable", "disable"],),
                "early_stopping": (["enable", "disable"],),
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
                "self_recursive": (["enable", "disable"],),
                "recursive_level": ("INT", {
                    "default": 0,
                    "min":0,
                    "max":50,
                    "step":1
                }),
            },
        }
    
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "generate"

    CATEGORY = "Prompt Generator"

    def RemoveDuplicates(self, line : str) -> list:
        prompts = line.split(",")
        pure_prompts = []
        can_add = True

        for prompt in prompts:
            keyword = prompt.strip("(),")
            for pos_prompt in pure_prompts:
                if keyword in pos_prompt:
                    can_add = False
                    break
            
            if keyword == "":
                can_add = False

            if can_add is False:
                continue
            pure_prompts.append(prompt)
            can_add = True

        return pure_prompts


    def Preprocess(self, line : str) -> str:
        from re import sub, compile
        
        pattern = compile(r'(,\s){2,}')

        temp_line = line.replace(u'\xa0', u' ')
        temp_line = temp_line.replace("\n", ", ")
        temp_line = temp_line.replace("  ", " ")
        temp_line = temp_line.replace("\t", " ")
        temp_line = sub(pattern, ', ', temp_line)

        # remove duplicates
        temp_line = ', '.join(self.RemoveDuplicates(temp_line))

        return temp_line
    
    def GetGeneratedText(self, generator, gen_args, seed, is_recursive, recursive_level):
        result = generator.generate_text(seed, gen_args)
        generated_text = self.Preprocess(seed + result.text)

        if is_recursive:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text = self.Preprocess(result.text)
            generated_text = self.Preprocess(seed + generated_text)
        else:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text += result.text
                generated_text = self.Preprocess(generated_text)
            
        return generated_text
    
    def LogOutputs(self, seed : str, generated_text : str, self_recursive : str, recursive_level : int, gen_settings, log_filename : str, ) -> None:
        from datetime import datetime

        print_string = f"{'  PROMPT GENERATOR OUTPUT  '.center(200, '#')}\n{generated_text}\n{'#'*200}\n"
        print(print_string)

        log_string = f"{'#'*200}\nDate & Time : {datetime.now()}\nSeed : {seed}\nPrompt : {generated_text}"
        log_string += f"min_token = {gen_settings.min_length}\n"
        log_string += f"max_token = {gen_settings.max_length}\n"
        log_string += f"do_sample = {gen_settings.do_sample}\n"
        log_string += f"early_stopping = {gen_settings.early_stopping}\n"
        log_string += f"num_beams = {gen_settings.num_beams}\n"
        log_string += f"temperature = {gen_settings.temperature}\n"
        log_string += f"top_k = {gen_settings.top_k}\n"
        log_string += f"top_p = {gen_settings.top_p}\n"
        log_string += f"no_repeat_ngram_size = {gen_settings.no_repeat_ngram_size}\n"
        log_string += f"self_recursive = {self_recursive}\nrecursive_level = {recursive_level}\n"
        with open(log_filename, "a") as file:
            file.write(log_string)

    def generate(self, clip, model_type, model_name, seed, min_token, max_token, do_sample, early_stopping, num_beams, temperature, top_k, top_p, no_repeat_ngram_size, self_recursive, recursive_level):
        from happytransformer import HappyGeneration, GENSettings
        from os.path import join, exists
        
        real_path = join(join("models", "prompt_generators"), model_name)
        prompt_log_filename = "generated_prompt.txt"

        if exists(prompt_log_filename) is False:
            file = open(prompt_log_filename, "w")
            file.close()

        generated_text = ""

        if exists(real_path) is False:
            print(f"{real_path} is not exists")
            generated_text = seed
        else:
            is_recursive = True if self_recursive == "enable" else False

            upper_model_type = model_type.upper()
            if model_type.find("/") != -1:
                upper_model_type = model_type.split("/")[1].upper()

            generator = HappyGeneration(model_type=upper_model_type, model_name=model_type, load_path=real_path)

            gen_settings = GENSettings(
                min_length=min_token,
                max_length=max_token,
                do_sample=True if do_sample == "enable" else False,
                early_stopping=True if early_stopping == "enable" else False,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                no_repeat_ngram_size=no_repeat_ngram_size
            )

            generated_text = self.GetGeneratedText(generator, gen_settings, seed, is_recursive, recursive_level)

        self.LogOutputs(seed, generated_text, self_recursive, recursive_level, gen_settings, prompt_log_filename)

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