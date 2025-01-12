"""
    Actually uninstalling packages is not suitable
    because they maybe need by other packages too
"""

"""
import subprocess
import sys
import platform
import generator.utility

common_packages = [
    "optimum",
    "optimum[onnxruntime-gpu]",
    "peft",
]


def uninstall_packages(packages: list[str]) -> None:
    for package in packages:
        print(f"/_\\ Uninstalling {package}")
        command = f"{sys.executable} -m pip uninstall --yes {package}"
        process = subprocess.run(command, shell=True, check=True, capture_output=True)

        if process.returncode != 0:
            print(f"{package} uninstallation is failed\nError: {process.stdout}")
"""


if __name__ == "__main__":
    print(" Prompt Generator ComfyUI Node ".center(100, "-"))
    print(
        "/_\\ If you have encountered issue(s) please feel free to open issue on repository's Github page"
    )

    """
    packages_to_uninstall = []
    packages_to_uninstall.extend(common_packages)

    os_name = platform.system()

    if os_name != "Linux" and generator.utility.check_torch_version_is_enough(2, 4):
        packages_to_uninstall.append("optimum-quanto")

    uninstall_packages()
    """

    print("/_\\ Uninstallation is compeleted successfully")
