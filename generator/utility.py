from os import listdir
from enum import Enum
from platform import system
from torch import __version__ as torch_version
from transformers import __version__ as transformers_version
from typing import Tuple
from functools import lru_cache
from importlib.util import find_spec
from comfy.model_management import get_torch_device, is_device_cuda


def check_required_package_version(
    current_major: int, current_minor: int, required_major: int, required_minor: int
) -> bool:
    return (current_major, current_minor) >= (required_major, required_minor)


def get_major_minor_versions(
    version_str: str, split_char: str = "."
) -> Tuple[int, int]:
    version_splitted = version_str.split(split_char)
    version_major = int(version_splitted[0])
    version_minor = int(version_splitted[1])

    return (version_major, version_minor)


@lru_cache
def check_package_version(
    version_str: str, min_major: int, min_minor: int, split_char: str = "."
) -> bool:
    package_major, package_minor = get_major_minor_versions(version_str, split_char)

    return check_required_package_version(
        package_major, package_minor, min_major, min_minor
    )


def check_torch_version_is_enough(min_major: int, min_minor: int) -> bool:
    return check_package_version(torch_version, min_major, min_minor)


def check_transformers_version(min_major: int, min_minor: int) -> bool:
    return check_package_version(transformers_version, min_major, min_minor)


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
    if is_device_cuda(get_torch_device()) and find_spec("bitsandbytes"):
        return QuantizationPackage.BITSANDBYTES

    if (
        check_torch_version_is_enough(2, 4)
        and find_spec("optimum")
        and find_spec("optimum.quanto")
    ):
        return QuantizationPackage.QUANTO

    return QuantizationPackage.NONE


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


def is_base_model(path: str) -> bool:
    files = listdir(path)

    for file in files:
        if "adapter" in file:
            return False

    return True
