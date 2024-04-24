from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
)

from transformers import pipeline as tf_pipe
from optimum.pipelines import pipeline as opt_pipe

from optimum.onnxruntime import ORTModelForCausalLM

from platform import system

from comfy.model_management import (
    get_torch_device,
    should_use_fp16,
    should_use_bf16,
    is_device_mps,
)
from torch import bfloat16 as torch_bfloat16
from torch import float16 as torch_float16
from torch import float32 as torch_float32

from torch import compile as torch_compile


def get_model(model_name: str, use_device_map: bool = True):
    dev = get_torch_device()

    if should_use_bf16(device=dev):
        req_torch_dtype = torch_bfloat16
    elif should_use_fp16(device=dev):
        req_torch_dtype = torch_float16
    else:
        req_torch_dtype = torch_float32

    if use_device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=req_torch_dtype
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device=dev, torch_dtype=req_torch_dtype
        )

    # torch.compile is supported only in Linux
    # has to be tested though
    if system() == "Linux":
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
    is_mps = is_device_mps(get_torch_device())

    if is_mps:
        model = get_model(model_name, use_device_map=False)
    else:
        model = get_model(model_name, use_device_map=True)

    tokenizer = get_tokenizer(model_name)

    pipe = opt_pipe(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        accelerator="bettertransformer",
        framework="pt",
        device=get_torch_device() if is_mps else None,
    )

    return pipe
