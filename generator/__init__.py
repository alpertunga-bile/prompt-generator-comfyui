import sys
from os.path import dirname
from subprocess import run
from importlib.util import find_spec
from platform import system
from torch import __version__ as torch_version

os_name = system()

sys.path.append(dirname(__file__))


def check_package(package_name: str, install_name: str) -> None:
    if find_spec(package_name):
        return

    print(f"/_\\ Installing {package_name}")

    command = f"{sys.executable} -m pip install {install_name}"

    process = run(command, shell=True, check=True, capture_output=True)

    if process.returncode != 0:
        print(f"{package_name} installation is failed\nError: {process.stdout}")


print(" Prompt Generator ComfyUI Node ".center(100, "-"))

# Check required packages
print("/_\\ Checking packages")

check_package("transformers", "transformers")
check_package("accelerate", "accelerate")

# triton package exists only in Linux
if os_name == "Linux":
    check_package("triton", "triton")

check_package("optimum", "optimum")
check_package("onnxruntime", "optimum[onnxruntime-gpu]")

# use_fast for tokenizers used this
check_package("sentencepiece", "transformers[sentencepiece]")

check_package("peft", "peft")


def check_torch_version_is_enough(min_major: int, min_minor: int) -> bool:
    torch_version_splitted = torch_version.split(".")
    torch_version_major = int(torch_version_splitted[0])
    torch_version_minor = int(torch_version_splitted[1])

    major_check = torch_version_major >= min_major
    minor_check = torch_version_major == min_major and torch_version_minor >= min_minor

    if major_check or minor_check:
        return True
    else:
        return False


if os_name == "Linux":
    check_package("bitsandbytes", "bitsandbytes")
elif check_torch_version_is_enough(2, 4):
    check_package("optimum-quanto", "optimum-quanto")
