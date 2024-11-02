from os import listdir
from enum import Enum
from platform import system
from torch import __version__ as torch_version
from transformers import __version__ as transformers_version
from typing import Tuple
from functools import lru_cache


class ModelType(Enum):
    DEFAULT = 1
    ONNX = 2
    BETTERTRANSFORMER = 3
    NONE = 4


@lru_cache
def check_required_package_version(
    current_major: int, current_minor: int, required_major: int, required_minor: int
) -> bool:
    major_check = current_major >= required_major
    minor_check = current_major == required_major and current_minor >= required_minor

    return major_check or minor_check


@lru_cache
def get_major_minor_versions(
    version_str: str, split_char: str = "."
) -> Tuple[int, int]:
    version_splitted = version_str.split(split_char)
    version_major = int(version_splitted[0])
    version_minor = int(version_splitted[1])

    return (version_major, version_minor)


def check_torch_version_is_enough(min_major: int, min_minor: int) -> bool:
    torch_version_major, torch_version_minor = get_major_minor_versions(torch_version)

    return check_required_package_version(
        torch_version_major, torch_version_minor, min_major, min_minor
    )


def check_transformers_version(min_major: int, min_minor: int) -> bool:
    transformers_version_major, transformers_version_minor = get_major_minor_versions(
        transformers_version
    )

    return check_required_package_version(
        transformers_version_major, transformers_version_minor, min_major, min_minor
    )


class QuantizationPackage(Enum):
    NONE = 1
    QUANTO = 2
    BITSANDBYTES = 3


class QuantizationType(Enum):
    NONE = 1
    EightBit = 2
    FourBit = 3
    EightFloat = 4


def get_quantization_package() -> QuantizationPackage:
    if system() == "Linux":
        return QuantizationPackage.BITSANDBYTES
    elif check_torch_version_is_enough(2, 4):
        return QuantizationPackage.QUANTO
    else:
        return QuantizationPackage.NONE


@lru_cache
def get_usable_quantize_sizes() -> list[str]:
    quant_package = get_quantization_package()
    quant_sizes = ["none"]

    if quant_package == QuantizationPackage.BITSANDBYTES:
        quant_sizes = quant_sizes + ["int8", "int4"]
    elif quant_package == QuantizationPackage.QUANTO:
        quant_sizes = quant_sizes + ["int8", "float8", "int4"]

    return quant_sizes


@lru_cache
def str_to_quant_type(type_str: str) -> QuantizationType:
    quantize_type = QuantizationType.NONE

    if type_str == "int8":
        quantize_type = QuantizationType.EightBit
    elif type_str == "int4":
        quantize_type = QuantizationType.FourBit
    elif type_str == "float8":
        quantize_type = QuantizationType.EightFloat

    return quantize_type


def get_variable_dictionary(given_class) -> dict:
    return {
        key: value
        for key, value in given_class.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }


@lru_cache
def is_base_model(path: str) -> bool:
    files = listdir(path)

    for file in files:
        if "adapter" in file:
            return False

    return True


@lru_cache
def get_accelerator_type(path: str) -> ModelType:
    files = listdir(path)
    accelerator_type = ModelType.NONE

    for file in files:
        if file.endswith(".onnx"):
            accelerator_type = ModelType.ONNX
            break
        if file.endswith(".bin") or file.endswith(".safetensors"):
            accelerator_type = ModelType.BETTERTRANSFORMER
            break

    return accelerator_type
