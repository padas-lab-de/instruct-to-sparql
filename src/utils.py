import io
import json
import os
from numbers import Number
from typing import TYPE_CHECKING, List

import math
import torch
from accelerate.utils import is_deepspeed_available
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import logging
load_dotenv()
from data.nl_generation.utils import OpenAIDecodingArguments, openai_completion

if is_deepspeed_available():
    pass

if TYPE_CHECKING:
    pass

MODELS_DIR = "/data/llms/"
DATASET_DIR = "/data/datasets/"
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

def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def get_generator_fn(model_name, path):
    def generator_fn(messages, **generation_kwargs):
        decoding_args = OpenAIDecodingArguments(
            **generation_kwargs
        )
        if isinstance(messages, List) and isinstance(messages[0], List):
            generated_texts = []
            if os.path.exists(f"{path}/generations_partial.json"):
                generated_texts = jload(f"{path}/generations_partial.json")
            try:
                for i, message in tqdm(enumerate(messages), desc="Generating texts with OpenAI", total=len(messages)):
                    if i < len(generated_texts):
                        continue
                    generated_text = openai_completion(message, model_name=model_name, decoding_args=decoding_args,
                                                       return_text=True)
                    generated_texts.append(generated_text)
                return generated_texts
            except Exception or KeyboardInterrupt as e:
                jdump(generated_texts, f"{path}/generations_partial.json")
                raise e
        else:
            generated_texts = openai_completion(messages, model_name=model_name, decoding_args=decoding_args,
                                           return_text=True)
        return [generated_texts]

    return generator_fn
