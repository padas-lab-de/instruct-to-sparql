import os
from numbers import Number
from typing import TYPE_CHECKING, List

import math
import torch
from accelerate.utils import is_deepspeed_available
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import logging, pipeline, AutoModelForCausalLM, AutoTokenizer
load_dotenv()
from data.nl_generation.utils import OpenAIDecodingArguments, openai_completion

if is_deepspeed_available():
    pass

if TYPE_CHECKING:
    pass

MODELS_DIR = "/data/llms/"
os.makedirs(MODELS_DIR, exist_ok=True)

# Set environment variables
os.environ["TORCHELASTIC_ERROR_FILE"] = "torchelastic_error_file.txt"
os.environ["WANDB_DIR"] = "/data/wandb"

logger = logging.get_logger(__name__)


def set_seed(seed: int = 19980406):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def significant(x: Number, ndigits=2) -> Number:
    """
    Cut the number up to its `ndigits` after the most significant
    """
    if isinstance(x, torch.Tensor):
        x = x.item()

    if not isinstance(x, Number) or math.isnan(x) or x == 0:
        return x

    return round(x, ndigits - int(math.floor(math.log10(abs(x)))))


def get_generator_fn(model_path, model_kwargs=None, tokenizer_kwargs=None, huggingface=True):
    if huggingface:
        model_kwargs = model_kwargs or {}
        tokenizer_kwargs = tokenizer_kwargs or {}
        # Generate text based on the prompt
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if "Llama-3" in model_path:
            model.half()
        tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
        if tokenizer.padding_side == "right":
            tokenizer.padding_side = "left"
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        def generator_fn(messages, **generation_kwargs):
            generation_kwargs.update({"tokenizer": tokenizer, "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id})
            return pipe(messages, **generation_kwargs)
    else:
        def generator_fn(messages, **generation_kwargs):
            decoding_args = OpenAIDecodingArguments(
                **generation_kwargs
            )
            if isinstance(messages, List) and isinstance(messages[0], List):
                generated_texts = []
                for message in tqdm(messages, desc="Generating texts", total=len(messages)):
                    generated_text = openai_completion(message, model_name=model_path, decoding_args=decoding_args,
                                                       return_text=True)
                    generated_texts.append(generated_text)
                return generated_texts
            else:
                generated_texts = openai_completion(messages, model_name=model_path, decoding_args=decoding_args,
                                               return_text=True)
            return [generated_texts]

    return generator_fn
