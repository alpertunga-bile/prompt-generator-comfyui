from os import listdir
from os.path import join, isdir, exists
from preprocess import preprocess
from generator.generate import GenerateArgs, Generator
from folder_paths import models_dir, base_path
from datetime import date, datetime


class PromptGenerator:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "model_name": (
                    [
                        file
                        for file in listdir(join(models_dir, "prompt_generators"))
                        if isdir(join(models_dir, "prompt_generators", file))
                    ],
                ),
                "accelerate": (["enable", "disable"],),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "((masterpiece, best quality, ultra detailed)), illustration, digital art, 1girl, solo, ((stunningly beautiful))",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "min_new_tokens": (
                    "INT",
                    {"default": 20, "min": 0, "max": 100, "step": 1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 50, "min": 35, "max": 200, "step": 1},
                ),
                "do_sample": (["disable", "enable"],),
                "early_stopping": (["disable", "enable"],),
                "num_beams": ("INT", {"default": 5, "min": 1, "max": 50, "step": 1}),
                "num_beam_groups": (
                    "INT",
                    {"default": 1, "min": 0, "max": 50, "step": 1},
                ),
                "diversity_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 150, "step": 1}),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.1},
                ),
                "no_repeat_ngram_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "remove_invalid_values": (["disable", "enable"],),
                "self_recursive": (["disable", "enable"],),
                "recursive_level": (
                    "INT",
                    {"default": 0, "min": 0, "max": 50, "step": 1},
                ),
                "preprocess_mode": (["exact_keyword", "exact_prompt", "none"],),
            },
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
    )
    RETURN_NAMES = (
        "first_output",
        "second_output",
        "third_output",
        "fourth_output",
        "fifth_output",
    )

    FUNCTION = "generate"

    CATEGORY = "Prompt Generator"

    def get_generated_texts(
        self,
        generator: Generator,
        gen_args: GenerateArgs,
        prompt: str,
        is_self_recursive: bool,
        recursive_level: int,
        preprocess_mode: str,
    ) -> list[str]:
        results = generator.generate_multiple_output_texts(prompt, gen_args)
        gen_texts = []

        for result in results:
            generated_text = preprocess(prompt + result, preprocess_mode)

            if is_self_recursive:
                for _ in range(0, recursive_level):
                    result = generator.generate_text(generated_text, gen_args)
                    generated_text = preprocess(result, preprocess_mode)
                generated_text = preprocess(prompt + generated_text, preprocess_mode)
            else:
                for _ in range(0, recursive_level):
                    result = generator.generate_text(generated_text, gen_args)
                    generated_text += result
                    generated_text = preprocess(generated_text, preprocess_mode)

            gen_texts.append(generated_text)

        return gen_texts

    def log_outputs(
        self,
        model_name: str,
        prompt: str,
        generated_texts: list[str],
        self_recursive: str,
        recursive_level: int,
        preprocess_mode: str,
        gen_settings: GenerateArgs,
        log_filename: str,
    ) -> None:
        print_string = f"{'  PROMPT GENERATOR OUTPUT  '.center(200, '#')}\n"

        for i in range(len(generated_texts)):
            print_string += f"[{i + 1}. Prompt] {generated_texts[i]}\n{'-'*200}\n"
        print_string += f"{'#'*200}\n"

        print(print_string)

        with open(log_filename, "a") as file:
            file.write(f"{'#'*200}\n")
            file.write(f"Date & Time           : {datetime.now()}\n")
            file.write(f"Model                 : {model_name}\n")
            file.write(f"Prompt                : {prompt}\n")
            file.write(f"Generated Prompts     :\n")

            for i in range(len(generated_texts)):
                file.write(
                    f"[{i + 1}. Prompt]           : {generated_texts[i]}\n{'-'*200}\n"
                )

            file.write(f"cfg                   : {gen_settings.guidance_scale}\n")
            file.write(f"min_new_tokens        : {gen_settings.min_new_tokens}\n")
            file.write(f"max_new_tokens        : {gen_settings.max_new_tokens}\n")
            file.write(f"do_sample             : {gen_settings.do_sample}\n")
            file.write(f"early_stopping        : {gen_settings.early_stopping}\n")
            file.write(f"early_stopping        : {gen_settings.early_stopping}\n")
            file.write(f"num_beams             : {gen_settings.num_beams}\n")
            file.write(f"num_beam_groups       : {gen_settings.num_beam_groups}\n")
            file.write(f"temperature           : {gen_settings.temperature}\n")
            file.write(f"top_k                 : {gen_settings.top_k}\n")
            file.write(f"top_p                 : {gen_settings.top_p}\n")
            file.write(f"repetition_penalty    : {gen_settings.repetition_penalty}\n")
            file.write(f"no_repeat_ngram_size  : {gen_settings.no_repeat_ngram_size}\n")
            file.write(
                f"remove_invalid_values : {gen_settings.remove_invalid_values}\n"
            )
            file.write(f"self_recursive        : {self_recursive}\n")
            file.write(f"recursive_level       : {recursive_level}\n")
            file.write(f"preprocess_mode       : {preprocess_mode}\n")

    def tokenize_texts(self, clip, texts: list[str]) -> list:
        processed = []

        for text in texts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            processed.append([[cond, {"pooled_output": pooled}]])

        return processed

    def generate(
        self,
        clip,
        model_name,
        accelerate,
        prompt,
        cfg,
        min_new_tokens,
        max_new_tokens,
        do_sample,
        early_stopping,
        num_beams,
        num_beam_groups,
        diversity_penalty,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        remove_invalid_values,
        self_recursive,
        recursive_level,
        preprocess_mode,
    ):
        root = join(models_dir, "prompt_generators")
        real_path = join(root, model_name)
        prompt_log_filename = (
            join(base_path, "generated_prompts", str(date.today())) + ".txt"
        )

        if exists(prompt_log_filename) is False:
            file = open(prompt_log_filename, "w")
            file.close()

        if exists(real_path) is False:
            raise ValueError(f"{real_path} is not exists")

        is_self_recursive = True if self_recursive == "enable" else False
        is_accelerate = True if accelerate == "enable" else False

        generator = Generator(real_path, is_accelerate)

        gen_settings = GenerateArgs(
            guidance_scale=cfg,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=True if do_sample == "enable" else False,
            early_stopping=True if early_stopping == "enable" else False,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            remove_invalid_values=True if remove_invalid_values == "enable" else False,
        )

        generated_texts = self.get_generated_texts(
            generator,
            gen_settings,
            prompt,
            is_self_recursive,
            recursive_level,
            preprocess_mode,
        )

        self.log_outputs(
            model_name,
            prompt,
            generated_texts,
            self_recursive,
            recursive_level,
            preprocess_mode,
            gen_settings,
            prompt_log_filename,
        )

        return tuple(self.tokenize_texts(clip, generated_texts))
