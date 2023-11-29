from sys import path
from os.path import dirname, exists
from subprocess import run
from importlib.util import find_spec
from platform import system

os_name = system()

path.append(dirname(__file__))


def check_package(package_name: str, install_name: str) -> None:
    if find_spec(package_name):
        return

    print(f"/_\ Installing {package_name}")

    # check if portable version or manual
    if exists("python_embeded"):
        command = f".\\python_embeded\\python.exe -s -m pip install {install_name}"
    else:
        command = f"pip install {install_name}"

    process = run(command, shell=True, check=True, capture_output=True)


print(" Prompt Generator ComfyUI Node ".center(100, "-"))

# Check required packages
print("/_\ Checking packages")

check_package("transformers", "transformers")
check_package("accelerate", "accelerate")

# triton package exists only in Linux
if os_name == "Linux":
    check_package("triton", "triton")

check_package("optimum", "optimum")
check_package("onnxruntime", "optimum[onnxruntime-gpu]")

"""
# This package is for onnx models that run on CPU
if find_spec("onnxruntime") is None:
    print(f"/_\ Installing onnxruntime")
    process = run(
        f"pip install optimum[onnxruntime]", shell=True, check=True, capture_output=True
    )
"""
