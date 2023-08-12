from dataclasses import dataclass
from generator.model import get_onnx_pipeline, get_default_pipeline
from generator.utility import get_accelerator_type, get_variable_dictionary
from transformers import Pipeline


@dataclass
class GenerateArgs:
    num_return_sequences: int = 1
    return_full_text: bool = False
    min_length: int = 0
    max_length: int = 50
    early_stopping: bool = False
    do_sample: bool = False
    num_beams: int = 1
    num_beam_groups: int = 1
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

    def __init__(self, model_path: str) -> None:
        accelerator_type = get_accelerator_type(model_path)

        if accelerator_type == "onnx":
            self.pipe = get_onnx_pipeline(model_name=model_path)
        elif accelerator_type == "bettertransformer":
            self.pipe = get_default_pipeline(model_name=model_path)
        else:
            raise ValueError(
                "Cant define accelerator type by folder. Can't find .onnx file for onnx, .bin for bettertransformer. Please check your model"
            )

    def generate_text(
        self,
        input: str,
        args: GenerateArgs = GenerateArgs(),
    ) -> str:
        if self.pipe is None:
            raise RuntimeError("Pipeline is NONE. Please check your model path")

        args = get_variable_dictionary(args)
        output = self.pipe(input, **args)

        return output[0]["generated_text"]
