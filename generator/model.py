from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
)

from transformers import pipeline as tf_pipe
from optimum.pipelines import pipeline as opt_pipe

from optimum.onnxruntime import ORTModelForCausalLM

from comfy.model_management import is_device_mps, get_torch_device


def get_model(model_name: str):
    from platform import system

    """
    mps is not supporting bfloat16 and the model weights may be bfloat16
    but torch_dtype is not considering it
    """
    if is_device_mps(get_torch_device()):
        from torch import float16 as torch_float16
        from torch import float32 as torch_float32

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch_float16
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype=torch_float32
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
        )

    # torch.compile is supported only in Linux
    # has to be tested though
    if system() == "Linux":
        from torch import compile as torch_compile

        torch_compile(model)

    return model


def get_tokenizer(model_name: str):
    # use the fast implementation of the tokenizer if possible
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    return tokenizer


def get_default_pipeline(model_name: str) -> Pipeline:
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    pipe = tf_pipe(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
    )

    return pipe


def get_onnx_pipeline(model_name: str, is_native: bool = True) -> Pipeline:
    if is_native:
        model = ORTModelForCausalLM.from_pretrained(model_name)
    else:
        model = ORTModelForCausalLM.from_pretrained(model_name, export=True)

    tokenizer = get_tokenizer(model_name)

    pipe = opt_pipe(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        accelerator="ort",
        framework="pt",
    )

    return pipe


def get_bettertransformer_pipeline(model_name: str) -> Pipeline:
    model = get_model(model_name)
    tokenizer = get_tokenizer(model_name)

    pipe = opt_pipe(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        accelerator="bettertransformer",
        framework="pt",
    )

    return pipe
