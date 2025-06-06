from comfy.model_management import get_torch_device
from generator.model import get_model_tokenizer

from generator.utility import (
    get_accelerator_type,
    get_variable_dictionary,
    str_to_quant_type,
    check_transformers_version,
    ModelType,
)
from generator.preprocess import preprocess

from dataclasses import dataclass


@dataclass
class GenerateArgs:
    num_return_sequences: int = 1
    min_new_tokens: int = 0
    max_new_tokens: int = 50
    early_stopping: bool = False
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.8
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 0
    remove_invalid_values: bool = False
    guidance_scale: float = 1.0


@dataclass
class Generator:
    model = None
    tokenizer = None
    dev = None
    extra_params = {}

    def __init__(
        self,
        model_path: str,
        is_accelerate: bool,
        is_token_healing: bool,
        model_quant_type: str,
    ) -> None:
        quantize_type = str_to_quant_type(model_quant_type)

        if is_accelerate is False:
            self.model, self.tokenizer = get_model_tokenizer(
                model_path, ModelType.DEFAULT, quantize_type
            )
        else:
            accelerator_type = get_accelerator_type(model_path)

            self.model, self.tokenizer = get_model_tokenizer(
                model_path, accelerator_type, quantize_type
            )

        self.tokenizer.padding_side = "left"

        self.extra_params["renormalize_logits"] = True
        self.extra_params["pad_token_id"] = self.tokenizer.eos_token_id

        if check_transformers_version(4, 6) and is_token_healing is True:
            self.extra_params["token_healing"] = True
            self.extra_params["tokenizer"] = self.tokenizer

        self.dev = get_torch_device()

    # generate single output
    def generate_text(
        self,
        input: str,
        args: GenerateArgs = GenerateArgs(),
    ) -> str:
        if self.model and self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer is NONE. Please check your model path"
            )

        args.num_return_sequences = 1
        given_input = input

        if "token_healing" in self.extra_params:
            given_input = input.strip()

        inputs = self.tokenizer(
            given_input, padding=True, truncation=True, return_tensors="pt"
        ).to(self.dev)

        generated_ids = self.model.generate(
            **inputs,
            **get_variable_dictionary(args),
            **self.extra_params,
        )

        output = self.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True, cleanup_tokenization_spaces=True
        )

        return output

    # generate 5 outputs
    def generate_multiple_texts(
        self,
        input: str,
        args: GenerateArgs = GenerateArgs(),
    ) -> list[str]:
        if self.model and self.tokenizer is None:
            raise RuntimeError(
                "Model and tokenizer is NONE. Please check your model path"
            )

        args.num_return_sequences = 5
        given_input = input

        if "token_healing" in self.extra_params:
            given_input = input.strip()

        inputs = self.tokenizer(
            given_input, padding=True, truncation=True, return_tensors="pt"
        ).to(self.dev)

        generated_ids = self.model.generate(
            **inputs,
            **get_variable_dictionary(args),
            **self.extra_params,
        )

        outputs = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        return outputs


"""
    first generating 5 outputs
    then for each output doing the recursion if specified
"""


def get_generated_texts(
    generator: Generator,
    gen_args: GenerateArgs,
    prompt: str,
    is_self_recursive: bool,
    recursive_level: int,
    preprocess_mode: str,
) -> list[str]:
    results = generator.generate_multiple_texts(prompt, gen_args)
    gen_texts = []

    for result in results:
        generated_text = preprocess(", ".join([prompt, result]), preprocess_mode)

        if recursive_level == 0:
            gen_texts.append(generated_text)
            continue

        if is_self_recursive:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text = preprocess(result, preprocess_mode)

            """
                in self recursive, generated_text is not include the prompt string
            """
            generated_text = preprocess(
                ", ".join([prompt, generated_text]), preprocess_mode
            )
        else:
            for _ in range(0, recursive_level):
                result = generator.generate_text(generated_text, gen_args)
                generated_text = preprocess(
                    ", ".join([generated_text, result]), preprocess_mode
                )

        gen_texts.append(generated_text)

    return gen_texts


__all__ = [GenerateArgs, Generator, get_generated_texts]
