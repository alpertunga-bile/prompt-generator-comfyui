from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from optimum.bettertransformer import BetterTransformer
from optimum.onnxruntime import ORTModelForCausalLM

from torch import bfloat16 as torch_bfloat16
from torch import float16 as torch_float16
from torch import float32 as torch_float32
from torch import compile as torch_compile

from platform import system

from comfy.model_management import (
    get_torch_device,
    should_use_fp16,
    should_use_bf16,
)

from .utility import (
    ModelType,
    QuantizationType,
    QuantizationPackage,
    get_quantization_package,
)


def get_torch_dtype():
    dev = get_torch_device()

    if should_use_bf16(device=dev):
        req_torch_dtype = torch_bfloat16
    elif should_use_fp16(device=dev):
        req_torch_dtype = torch_float16
    else:
        req_torch_dtype = torch_float32

    return req_torch_dtype


def get_quanto_config(type: QuantizationType):
    from transformers import QuantoConfig

    quanto_config = None

    if type == QuantizationType.EightBit:
        quanto_config = QuantoConfig(weights="int8")
    elif type == QuantizationType.FourBit:
        quanto_config = QuantoConfig(weight="int4")
    elif type == QuantizationType.EightFloat:
        quanto_config = QuantoConfig(weights="float8")

    return quanto_config


def get_bitsandbytes_config(type: QuantizationType):
    from transformers import BitsAndBytesConfig

    bnb_config = None

    if type == QuantizationType.EightBit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif type == QuantizationType.FourBit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=get_torch_dtype(),
        )

    return bnb_config


def get_quantization_config(type: QuantizationType):
    quant_config = None

    quant_package = get_quantization_package()

    if quant_package == QuantizationPackage.QUANTO:
        quant_config = get_quanto_config(type)
    elif quant_package == QuantizationPackage.BITSANDBYTES:
        quant_config = get_bitsandbytes_config(type)

    return quant_config


def get_model(model_name: str, type: QuantizationType, use_device_map: bool = True):
    req_torch_dtype = get_torch_dtype()
    quant_config = get_quantization_config(type)

    if use_device_map:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=req_torch_dtype,
            quantization_config=quant_config,
        )
    else:
        dev = get_torch_device()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device=dev,
            torch_dtype=req_torch_dtype,
            quantization_config=quant_config,
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

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return tokenizer


def get_model_tokenizer(
    model_path: str,
    type: ModelType,
    quant_type: QuantizationType,
    is_native: bool = True,
):
    if type == ModelType.ONNX:
        if is_native:
            model = ORTModelForCausalLM.from_pretrained(model_path)
        else:
            model = ORTModelForCausalLM.from_pretrained(model_path, export=True)
    elif type == ModelType.BETTERTRANSFORMER or ModelType.DEFAULT:
        # is_mps = is_device_mps(get_torch_device())

        model = get_model(model_path, quant_type, use_device_map=True)

    if type == ModelType.BETTERTRANSFORMER:
        try:
            model = BetterTransformer.transform(model)
        except:
            pass

    tokenizer = get_tokenizer(model_path)

    return (model, tokenizer)
