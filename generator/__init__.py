from sys import path
from os.path import dirname
from subprocess import run
from importlib.util import find_spec
from platform import system

os_name = system()

path.append(dirname(__file__))


def check_package(package_name: str, install_name: str) -> None:
    if find_spec(package_name) is None:
        print(f"/_\ Installing {package_name}")
        process = run(
            f"pip install {install_name}", shell=True, check=True, capture_output=True
        )


# Check required packages
print("/_\ Checking packages")

check_package("transformers", "transformers")
check_package("accelerate", "accelerate")

if os_name == "Linux":
    check_package("triton", "triton")

check_package("optimum", "optimum")
check_package("onnxruntime-gpu", "optimum[onnxruntime-gpu]")

"""
# This package is for onnx models that run on CPU
if find_spec("onnxruntime") is None:
    print(f"/_\ Installing onnxruntime")
    process = run(
        f"pip install optimum[onnxruntime]", shell=True, check=True, capture_output=True
    )
"""
