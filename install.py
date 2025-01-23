import subprocess
import sys
import platform
import generator.utility

common_packages = [
    "transformers",
    "accelerate",
    "optimum",
    "optimum[onnxruntime-gpu]",
    "transformers[sentencepiece]",
    "peft",
]


def install_packages(packages: list[str]) -> None:
    for package in packages:
        print(f"/_\\ Installing {package}")
        command = f"{sys.executable} -m pip install {package}"
        process = subprocess.run(command, shell=True, check=True, capture_output=True)

        if process.returncode != 0:
            print(f"{package} installation is failed\nError: {process.stdout}")


if __name__ == "__main__":
    print(" Prompt Generator ComfyUI Node ".center(100, "-"))
    print("/_\\ Thank you for installing Prompt Generator node")

    packages_to_install = []
    packages_to_install.extend(common_packages)

    os_name = platform.system()

    if os_name == "Linux":
        packages_to_install.append("triton")
        packages_to_install.append("bitsandbytes")

    if os_name != "Linux" and generator.utility.check_torch_version_is_enough(2, 4):
        packages_to_install.append("optimum-quanto")

    install_packages(packages_to_install)

    print("/_\\ Installation is compeleted successfully")
    print("-" * 100)
