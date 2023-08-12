from os import listdir


def get_variable_dictionary(given_class) -> dict:
    return {
        key: value
        for key, value in given_class.__dict__.items()
        if not key.startswith("__") and not callable(key)
    }


def get_accelerator_type(path: str) -> str:
    files = listdir(path)
    accelerator_type = "none"

    for file in files:
        if file.endswith(".onnx"):
            accelerator_type = "onnx"
            break
        if file.endswith(".bin"):
            accelerator_type = "bettertransformer"
            break

    return accelerator_type
