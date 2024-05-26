from os import listdir
from enum import Enum
from platform import system
from torch import __version__ as torch_version


class ModelType(Enum):
    DEFAULT = 1
    ONNX = 2
    BETTERTRANSFORMER = 3
    NONE = 4


def check_torch_version_is_enough(min_major: int, min_minor: int) -> bool:
    torch_version_splitted = torch_version.split(".")
    torch_version_major = int(torch_version_splitted[0])
    torch_version_minor = int(torch_version_splitted[1])

    if torch_version_major >= min_major and torch_version_minor >= min_minor:
        return True
    else:
        return False


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
    elif check_torch_version_is_enough(2, 2):
        return QuantizationPackage.QUANTO
    else:
        return QuantizationPackage.NONE


def get_usable_quantize_sizes() -> list[str]:
    quant_package = get_quantization_package()
    quant_sizes = ["none"]

    if quant_package == QuantizationPackage.BITSANDBYTES:
        quant_sizes = quant_sizes + ["int8", "int4"]
    elif quant_package == QuantizationPackage.QUANTO:
        quant_sizes = quant_sizes + ["int8", "float8", "int4"]

    return quant_sizes


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


def is_base_model(path: str) -> bool:
    files = listdir(path)

    for file in files:
        if "adapter" in file:
            return False

    return True


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
