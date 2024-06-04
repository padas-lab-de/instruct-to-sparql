
# Code adapted from https://github.com/chujiezheng/chat_templates and https://github.com/huggingface/trl/blob/main/trl/models/utils.py
from dataclasses import dataclass
import trl.models.utils


@dataclass
class ChatMlAdaptedFormat:
    """Dataclass for special tokens used in ChatML, including system, user, assistant, bos, eos, and pad tokens."""

    def __init__(self, tokenizer=None):
        self.bos_token = tokenizer.bos_token if tokenizer else "<|im_start|>"
        self.eos_token = tokenizer.eos_token if tokenizer else "<|im_end|>"
        self.pad_token = tokenizer.eos_token if tokenizer else "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def response_template(self):
        return f"\n{self.assistant}\n Here is the SPARQL"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )

@dataclass
class MistralChatFormat:
    """Dataclass for special tokens used in Mistral Chat template, including system, user, assistant, bos, eos, and pad tokens."""
    bos_token = "<s>"
    eos_token = "</s>"
    pad_token = "</s>"
    @property
    def assistant(self):
        return f"[/INST]"

    @property
    def response_template(self):
        return f" [/INST] Here is the SPARQL"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = messages[0]['content'] | trim + '\n\n' %}"
            "{% set messages = messages[1:] %}"
            "{% else %}"
            "{% set system_message = '' %}"
            "{% endif %}"
            "{{ bos_token + '[INST] ' + system_message}}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] | trim + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + message['content'] | trim + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
@dataclass
class Llama2ChatFormat:
    """Dataclass for special tokens used in Llama Chat template, including system, user, assistant, bos, eos, and pad tokens."""
    bos_token = "<s>"
    eos_token = "</s>"
    pad_token = "</s>"
    @property
    def assistant(self):
        return f"[/INST]"

    @property
    def response_template(self):
        return f" [/INST] Here is the SPARQL"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set system_message = '<<SYS>>\n' + messages[0]['content'] | trim + '\n<</SYS>>\n\n' %}"
            "{% set messages = messages[1:] %}"
            "{% else %}"
            "{% set system_message = '' %}"
            "{% endif %}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% if loop.index0 == 0 %}"
            "{% set content = system_message + message['content'] %}"
            "{% else %}"
            "{% set content = message['content'] %}"
            "{% endif %}"
            "{% if message['role'] == 'user' %}"
            "{{ bos_token + '[INST] ' + content | trim + ' [/INST]' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ ' ' + content | trim + ' ' + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
        )
@dataclass
class Llama3ChatFormat:
    """Dataclass for special tokens used in Llama 3 Chat template, including system, user, assistant, bos, eos, and pad tokens."""
    bos_token = "<|begin_of_text|>"
    eos_token = "<|eot_id|>"
    pad_token = "<|eot_id|>"
    @property
    def assistant(self):
        return f"<|start_header_id|>assistant<|end_header_id|>\n\n"

    @property
    def response_template(self):
        return f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHere is the SPARQL"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set offset = 1 %}"
            "{% else %}"
            "{% set offset = 0 %}"
            "{% endif %}"
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}"
            "{{raise_exception('Conversation roles must alternate user/assistant/user/assistant/...')}}"
            "{% endif %}"
            "{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n'}}"
            "{% endif %}"
        )
@dataclass
class Phi3ChatFormat:
    """Dataclass for special tokens used in Phi 3 Chat template, including system, user, assistant, bos, eos, and pad tokens."""
    bos_token = "<|startoftext|>"
    eos_token = "<|endoftext|>"
    pad_token = "<|endoftext|>"
    @property
    def assistant(self):
        return f"<|assistant|>\n"

    @property
    def response_template(self):
        return f"\n<|assistant|>\nHere is the SPARQL"

    @property
    def chat_template(self):
        return (
            "{% if messages[0]['role'] == 'system' %}"
            "{% set offset = 1 %}"
            "{% else %}"
            "{% set offset = 0 %}"
            "{% endif %}"
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{{ '<|' + message['role'] + '|>\n' + message['content'] | trim + '<|end|>' + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|assistant|>\n' }}"
            "{% endif %}"
        )




trl.models.utils.FORMAT_MAPPING.update({
    "chatml": ChatMlAdaptedFormat,
    "mistral": MistralChatFormat,
    "llama": Llama2ChatFormat,
    "llama3": Llama3ChatFormat,
    "phi3": Phi3ChatFormat
})

FORMATS = trl.models.utils.FORMAT_MAPPING