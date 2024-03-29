from dataclasses import dataclass
from transformers import Pipeline

from generator.model import (
    get_default_pipeline,
    get_onnx_pipeline,
    get_bettertransformer_pipeline,
)
from generator.utility import get_accelerator_type, get_variable_dictionary
from generator.preprocess import preprocess


@dataclass
class GenerateArgs:
    num_return_sequences: int = 1
    return_full_text: bool = False
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
    pipe: Pipeline = None

    def __init__(self, model_path: str, is_accelerate: bool) -> None:
        if is_accelerate is False:
            self.pipe = get_default_pipeline(model_path)
            return

        accelerator_type = get_accelerator_type(model_path)

        if accelerator_type == "onnx":
            self.pipe = get_onnx_pipeline(model_name=model_path, is_native=True)
        elif accelerator_type == "bettertransformer":
            # onnx pipeline can broke easily so try without onnx pipeline
            self.try_wo_onnx_pipeline(model_path)
        else:
            raise ValueError(
                "Cant define the accelerator type by folder. Can't find .onnx file for onnx, .bin or .safetensors for bettertransformer and default pipeline. Please check your model"
            )

    # try with bettertransformer first then transformers
    def try_wo_onnx_pipeline(self, model_path: str):
        try:
            self.pipe = get_bettertransformer_pipeline(model_name=model_path)
        except:
            self.pipe = get_default_pipeline(model_path)

    # generate single output
    def generate_text(
        self,
        input: str,
        args: GenerateArgs = GenerateArgs(),
    ) -> str:
        if self.pipe is None:
            raise RuntimeError("Pipeline is NONE. Please check your model path")

        args.num_return_sequences = 1
        args = get_variable_dictionary(args)
        output = self.pipe(input, **args)

        return output[0]["generated_text"]

    # generate 5 outputs
    def generate_multiple_texts(
        self,
        input: str,
        args: GenerateArgs = GenerateArgs(),
    ) -> list[str]:
        if self.pipe is None:
            raise RuntimeError("Pipeline is NONE. Please check your model path")

        args.num_return_sequences = 5
        args = get_variable_dictionary(args)
        outputs = self.pipe(input, **args)

        return [output["generated_text"] for output in outputs]


# first generating 5 outputs
# then for each output doing the recursion if specified
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
