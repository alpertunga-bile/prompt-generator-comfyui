from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline

from transformers import pipeline as tf_pipe
from optimum.pipelines import pipeline as opt_pipe

from optimum.onnxruntime import ORTModelForCausalLM


def get_default_pipeline(model_name: str) -> Pipeline:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = tf_pipe(task="text-generation", model=model, tokenizer=tokenizer)

    return pipe


def get_onnx_pipeline(model_name: str) -> Pipeline:
    model = ORTModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = opt_pipe(
        task="text-generation", model=model, tokenizer=tokenizer, accelerator="ort"
    )

    return pipe


def get_bettertransformer_pipeline(model_name: str) -> Pipeline:
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pipe = opt_pipe(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        accelerator="bettertransformer",
    )

    return pipe
