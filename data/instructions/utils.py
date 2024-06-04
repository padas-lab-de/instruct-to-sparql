import copy
import dataclasses
import io
import json
# Todo: 1. Generate X re-formulated instructions for each query context + description using GPT3.5 and OpenAssistant.
# Todo: 2. Prepare the instructions for the LLMs with different templates.
import logging
import os
import time
from typing import Union, Sequence, Optional, Dict, List
import tiktoken

import openai  # for OpenAI API calls
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

client = openai.OpenAI(api_key=os.getenv("INFERENCE_API_KEY"), base_url=os.getenv("INFERENCE_API_URL"))


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def chat_completion_with_backoff(client, **kwargs):
    """Call openai.Completion.create() with exponential backoff."""
    try:
        return client.chat.completions.create(**kwargs)
    except openai.OpenAIError as e:
        logging.warning(f"OpenAIError: {e}.")
        if "Please reduce" in str(e):
            kwargs["max_tokens"] = int(kwargs["max_tokens"] * 0.8)
            logging.warning(f"Reducing target length to {kwargs['max_tokens']}, Retrying...")
            return chat_completion_with_backoff(client, **kwargs)
        else:
            raise e


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_tokens: int = 8192
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None


def openai_completion(
        messages: Union[List[Dict], List[Dict[str, str]], Dict[str, str]],
        decoding_args: OpenAIDecodingArguments,
        model_name="llama3",
        sleep_time=2,
        return_text=False,
        **decoding_kwargs,
):
    """Decode with OpenAI API.

    Args:
        messages: A string or a list of strings to complete. If it is a chat model the strings should be formatted
            as explained here: https://github.com/openai/openai-python/blob/main/chatml.md. If it is a chat model
            it can also be a dictionary (or list thereof) as explained here:
            https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
        decoding_args: Decoding arguments.
        model_name: Model name. Can be either in the format of "org/model" or just "model".
        sleep_time: Time to sleep once the rate-limit is hit.
        return_text: If True, return text instead of full completion object (which contains things like logprob).
        decoding_kwargs: Additional decoding arguments. Pass in `best_of` and `logit_bias` if you need them.

    Returns:
        A completion or a list of completions.
        Depending on return_text, return_openai_object, and decoding_args.n, the completion type can be one of
            - a string (if return_text is True)
            - an openai_object.OpenAIObject object (if return_text is False)
            - a list of objects of the above types (if decoding_args.n > 1)
    """

    is_prompt_str = isinstance(messages, str)
    if is_prompt_str:
        messages = [{"role": "user", "content": messages}]
    is_prompt_dict = isinstance(messages, dict)
    if is_prompt_dict:
        messages = [messages]

    batch_decoding_args = copy.deepcopy(decoding_args)  # cloning the decoding_args
    while True:
        try:
            shared_kwargs = dict(
                model=model_name,
                **batch_decoding_args.__dict__,
                **decoding_kwargs,
            )
            completion_chat = chat_completion_with_backoff(client, messages=messages, **shared_kwargs)
            choice = completion_chat.choices[0]

            completion = choice
            break
        except openai.OpenAIError as e:
            logging.warning(f"OpenAIError: {e}.")
            if "Please reduce" in str(e):
                batch_decoding_args.max_tokens = int(batch_decoding_args.max_tokens * 0.8)
                logging.warning(f"Reducing target length to {batch_decoding_args.max_tokens}, Retrying...")
            else:
                logging.warning("Hit request rate limit; retrying...")
                time.sleep(sleep_time)  # Annoying rate limit on requests.

    if return_text:
        completion = completion.message.content
    return completion


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
