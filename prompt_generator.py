from os import listdir
from os.path import join, isdir, exists
from torch import manual_seed
from torch.cuda import empty_cache
from gc import collect
from transformers import set_seed
from random import randint
from datetime import date

from generator.generate import GenerateArgs, Generator, get_generated_texts
from generator.utility import get_usable_quantize_sizes

from comfy.sd import CLIP
from folder_paths import models_dir, base_path

INT_MAX = 0xFFFFFFFFFFFFFFFF
FLOAT_MAX = 1_000_000.0


class PromptGenerator:
    _index = 0  # index to use for the cached generations, range in [0, 4]
    _generated_prompts = []  # last generated prompts
    _tokenized_prompts = []  # tokenized prompts from the last generated prompts
    _gen_settings = GenerateArgs  # gen configurations from the last generation

    @classmethod
    def INPUT_TYPES(s):
        quantize_sizes = get_usable_quantize_sizes()
        model_names = [
            file
            for file in listdir(join(models_dir, "prompt_generators"))
            if isdir(join(models_dir, "prompt_generators", file))
        ]

        return {
            "required": {
                "clip": ("CLIP",),
                "model_name": (model_names,),
                "accelerate": (["enable", "disable"],),
                "quantize": (quantize_sizes,),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "((masterpiece, best quality, ultra detailed)), illustration, digital art, 1girl, solo, ((stunningly beautiful)), ",
                    },
                ),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF},
                ),
                "lock": (["disable", "enable"],),
                "random_index": (["enable", "disable"],),
                "index": ("INT", {"default": 1, "min": 1, "max": 5}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": INT_MAX,
                        "step": 0.1,
                    },
                ),
                "min_new_tokens": (
                    "INT",
                    {"default": 20, "min": 0, "max": INT_MAX, "step": 1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 50, "min": 35, "max": INT_MAX, "step": 1},
                ),
                "do_sample": (["disable", "enable"],),
                "early_stopping": (["enable", "disable"],),
                "num_beams": (
                    "INT",
                    {"default": 1, "min": 1, "max": INT_MAX, "step": 1},
                ),
                "num_beam_groups": (
                    "INT",
                    {"default": 1, "min": 0, "max": INT_MAX, "step": 1},
                ),
                "diversity_penalty": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "top_k": ("INT", {"default": 50, "min": 0, "max": INT_MAX, "step": 1}),
                "top_p": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": FLOAT_MAX, "step": 0.1},
                ),
                "no_repeat_ngram_size": (
                    "INT",
                    {"default": 0, "min": 0, "max": INT_MAX, "step": 1},
                ),
                "remove_invalid_values": (["disable", "enable"],),
                "self_recursive": (["disable", "enable"],),
                "recursive_level": (
                    "INT",
                    {"default": 0, "min": 0, "max": FLOAT_MAX, "step": 1},
                ),
                "preprocess_mode": (["exact_keyword", "exact_prompt", "none"],),
            },
        }

    def __log_outputs(
        self,
        model_name: str,
        prompt: str,
        self_recursive: str,
        recursive_level: int,
        preprocess_mode: str,
        log_filename: str,
    ) -> None:
        print_string = f"{'  PROMPT GENERATOR OUTPUT  '.center(200, '#')}\n"

        print_string += f"Selected Prompt Index : {self._index + 1}\n\n"

        for i in range(len(self._generated_prompts)):
            print_string += (
                f"[{i + 1}. Prompt] {self._generated_prompts[i]}\n{'-'*200}\n"
            )
        print_string += f"{'#'*200}\n"

        print(print_string)

        from datetime import datetime

        with open(log_filename, "a") as file:
            file.write(f"{'#'*200}\n")
            file.write(f"Date & Time           : {datetime.now()}\n")
            file.write(f"Model                 : {model_name}\n")
            file.write(f"Prompt                : {prompt}\n")
            file.write(f"Generated Prompts     :\n")

            for i in range(len(self._generated_prompts)):
                file.write(
                    f"[{i + 1}. Prompt]           : {self._generated_prompts[i]}\n{'-'*200}\n"
                )
            file.write(f"Selected Prompt Index : {self._index + 1}\n")

            file.write(f"cfg                   : {self._gen_settings.guidance_scale}\n")
            file.write(f"min_new_tokens        : {self._gen_settings.min_new_tokens}\n")
            file.write(f"max_new_tokens        : {self._gen_settings.max_new_tokens}\n")
            file.write(f"do_sample             : {self._gen_settings.do_sample}\n")
            file.write(f"early_stopping        : {self._gen_settings.early_stopping}\n")
            file.write(f"num_beams             : {self._gen_settings.num_beams}\n")
            file.write(
                f"num_beam_groups       : {self._gen_settings.num_beam_groups}\n"
            )
            file.write(f"temperature           : {self._gen_settings.temperature}\n")
            file.write(f"top_k                 : {self._gen_settings.top_k}\n")
            file.write(f"top_p                 : {self._gen_settings.top_p}\n")
            file.write(
                f"repetition_penalty    : {self._gen_settings.repetition_penalty}\n"
            )
            file.write(
                f"no_repeat_ngram_size  : {self._gen_settings.no_repeat_ngram_size}\n"
            )
            file.write(
                f"remove_invalid_values : {self._gen_settings.remove_invalid_values}\n"
            )
            file.write(f"self_recursive        : {self_recursive}\n")
            file.write(f"recursive_level       : {recursive_level}\n")
            file.write(f"preprocess_mode       : {preprocess_mode}\n")

    def __tokenize_texts(self, clip: CLIP) -> list:
        processed = []

        # from nodes.py -> CLIPTextEncode -> encode
        for text in self._generated_prompts:
            tokens = clip.tokenize(text)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            processed.append([[cond, {"pooled_output": pooled}]])

        return processed

    RETURN_TYPES = (
        "CONDITIONING",
        "STRING",
    )
    RETURN_NAMES = ("gen_prompt", "gen_prompt_str")
    FUNCTION = "generate"
    CATEGORY = "Prompt Generator"

    def generate(
        self,
        clip: CLIP,
        model_name: str,
        accelerate: str,
        quantize: str,
        prompt: str,
        seed: int,
        lock: str,
        random_index: str,
        index: int,
        cfg: float,
        min_new_tokens: int,
        max_new_tokens: int,
        do_sample: str,
        early_stopping: str,
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        temperature: float,
        top_k: float,
        top_p: float,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        remove_invalid_values: str,
        self_recursive: str,
        recursive_level: int,
        preprocess_mode: str,
    ):
        # deal with encodings
        prompt = prompt.encode("ascii", "xmlcharrefreplace").decode()
        prompt = prompt.encode(errors="xmlcharrefreplace").decode()

        # create the prompt log file for current day
        prompt_log_filename = (
            join(base_path, "generated_prompts", str(date.today())) + ".txt"
        )

        is_do_sample = True if do_sample == "enable" else False

        # randint(min, max) -> [min, max]
        # index             -> [1, 5]
        self._index = randint(0, 4) if random_index == "enable" else index - 1

        is_lock_generation = True if lock == "enable" else False

        # check if it is the first generation with taking length of tokenized prompts
        # and the boolean with is lock enabled
        # if it is true just return from the lists with assigned new index (declaration is above)
        # log the outputs for the clearity
        if is_lock_generation is True and len(self._tokenized_prompts) > 0:
            self.__log_outputs(
                model_name,
                prompt,
                self_recursive,
                recursive_level,
                preprocess_mode,
                prompt_log_filename,
            )

            return (
                self._tokenized_prompts[self._index],
                self._generated_prompts[self._index],
            )

        # create relative path for the model
        model_path = join(models_dir, "prompt_generators", model_name)

        is_self_recursive = True if self_recursive == "enable" else False
        is_accelerate = True if accelerate == "enable" else False

        is_early_stopping = True if early_stopping == "enable" else False
        is_remove_invalid_values = True if remove_invalid_values == "enable" else False

        if is_do_sample:
            # huggingface supports [0, 2 ** 32 - 1] as seed
            set_seed(randint(0, 4294967294))
            manual_seed(seed)

        if exists(model_path) is False:
            raise ValueError(f"{model_path} is not exists")

        if exists(prompt_log_filename) is False:
            file = open(prompt_log_filename, "w")
            file.close()

        generator = Generator(model_path, is_accelerate, quantize)

        self._gen_settings = GenerateArgs(
            guidance_scale=cfg,
            min_new_tokens=min_new_tokens,
            max_new_tokens=max_new_tokens,
            do_sample=is_do_sample,
            early_stopping=is_early_stopping,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            remove_invalid_values=is_remove_invalid_values,
        )

        self._generated_prompts = get_generated_texts(
            generator,
            self._gen_settings,
            prompt,
            is_self_recursive,
            recursive_level,
            preprocess_mode,
        )

        self._tokenized_prompts = self.__tokenize_texts(clip)

        del generator
        empty_cache()
        collect()

        self.__log_outputs(
            model_name,
            prompt,
            self_recursive,
            recursive_level,
            preprocess_mode,
            prompt_log_filename,
        )

        return (
            self._tokenized_prompts[self._index],
            self._generated_prompts[self._index],
        )

    @classmethod
    def VALIDATE_INPUTS(s, clip: CLIP, model_name: str):
        model_path = join(models_dir, "prompt_generators", model_name)

        if not exists(model_path):
            return f"{model_path} is not exists"

        return True
