from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
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
    is_base_model,
    check_torch_version_is_enough
)

from peft import PeftModel


def get_torch_dtype():
    dev = get_torch_device()

    if should_use_bf16(device=dev):
        req_torch_dtype = torch_bfloat16
    elif should_use_fp16(device=dev):
        req_torch_dtype = torch_float16
    else:
        req_torch_dtype = torch_float32

    return req_torch_dtype


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


def get_model_from_base(model_name: str, required_torch_dtype, type: QuantizationType, is_acceleration: bool):
    model_configs = {}

    model_configs["device_map"] = "auto"
    model_configs["torch_dtype"] = required_torch_dtype
    
    quant_pack = get_quantization_package()

    if quant_pack == QuantizationPackage.BITSANDBYTES:
        quant_conf = get_bitsandbytes_config(type)

        model_configs["quantization_config"] = quant_conf

    if is_acceleration and check_torch_version_is_enough(2, 1):
        model_configs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_configs
    )

    if quant_pack == QuantizationPackage.QUANTO:
        from optimum.quanto import qfloat8, qint8, qint4, quantize, freeze

        if type == QuantizationType.EightBit:
            quantize(model, weights=qint8)
        elif type == QuantizationType.EightFloat:
            quantize(model, weights=qfloat8)
        elif type == QuantizationType.FourBit:
            quantize(model, weights=qint4)

        freeze(model)

    return model


def get_model_from_lora(model_name: str, required_torch_dtype, type: QuantizationType, is_acceleration: bool):
    model = get_model_from_base(model_name, required_torch_dtype, type, is_acceleration)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    model = PeftModel.from_pretrained(model, model_name, is_trainable=False)

    return model


def get_model(model_name: str, type: QuantizationType, is_acceleration: bool):
    req_torch_dtype = get_torch_dtype()

    if is_base_model(model_name):
        model = get_model_from_base(model_name, req_torch_dtype, type, is_acceleration)
    else:
        model = get_model_from_lora(model_name, req_torch_dtype, type, is_acceleration)

    """
        torch.compile is supported only in Linux
        has to be tested though
    """
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


def get_model_tokenizer(model_path: str, type: ModelType, quant_type: QuantizationType):
    """
        transformers -> ONNX operation brokes often
    """
    if type == ModelType.ONNX:
        model = ORTModelForCausalLM.from_pretrained(model_path)
    elif type == ModelType.BETTERTRANSFORMER:
        model = get_model(model_path, quant_type, is_acceleration=True)
    elif type == ModelType.DEFAULT:
        model = get_model(model_path, quant_type, is_acceleration=False)

    tokenizer = get_tokenizer(model_path)

    return (model, tokenizer)


__all__ = [get_model_tokenizer]
