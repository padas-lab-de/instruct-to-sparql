from datasets import load_dataset
from jinja2 import TemplateError


def load_data(name, subset: str = ""):
    """Load data from json file"""
    if subset:
        return load_dataset(name, name=subset)
    else:
        return load_dataset(name)


def get_formatting_func(tokenizer, formatting_style="chatml", annotated=False,
                        start_tag="[QUERY]", end_tag="[/QUERY]", split="train"):
    """Return formatting function based on the formatting style"""
    system_prompt = f"You are a helpful assistant. Your task is to provide a SPARQL query that answers the user's question. The query should be written between the tags {start_tag} and {end_tag}."
    if annotated:
        system_prompt = f"You are a WikiData chatbot. You will be given an instruction or a question. " \
                        "Convert the instruction or question into a SPARQL query. QIDs of entities and properties can be defined as text label.\n" \
                        f"Example: wdt:[property:instance of] instead of wdt:P31.\nThe query should be returned between the tags {start_tag} and {end_tag}."
    if annotated:
        output_key = "sparql_annotated"
    else:
        output_key = "sparql_query"
    answer_template = "Here is the SPARQL query:\n\n{start_tag}\n{query}\n{end_tag}\n"
    if formatting_style == "prompts":
        def formatting_prompts_func(example):
            output_texts = []
            label_texts = []
            for i in range(len(example[output_key])):
                text = f"{system_prompt}\n\n### Question: {example['instructions'][i]}\n"
                assistant_message = answer_template.format(query=example[output_key][i], start_tag=start_tag,
                                                           end_tag=end_tag)
                if split == "train":
                    text += f"### Answer: {assistant_message}"
                else:
                    text += f"### Answer: "
                output_texts.append(text)
                label_texts.append(assistant_message)
            if split == "train":
                return output_texts
            else:
                example["input_ids"] = tokenizer(output_texts,
                                                 padding=False, truncation=False,
                                                 add_special_tokens=False)["input_ids"]
                labels = example["input_ids"]
                example["labels"] = labels
                return example

        return formatting_prompts_func
    elif formatting_style == "chatml":
        def formatting_chatml_func(example):
            output_texts = []
            label_texts = []
            for i in range(len(example[output_key])):
                assistant_message = answer_template.format(query=example[output_key][i], start_tag=start_tag,
                                                           end_tag=end_tag)
                chat = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": example['instructions'][i]
                    }
                ]
                if split == "train":
                    chat.append(
                        {
                            "role": "assistant",
                            "content": assistant_message
                        }
                    )
                try:
                    output_texts.append(tokenizer.apply_chat_template(
                        chat, tokenize=False,
                        add_generation_prompt=False if split == "train" else True,
                    ))
                    label_texts.append(assistant_message)
                except TemplateError as e:
                    if "user/assistant/user/assistant/" in str(
                            e) or "Only user and assistant roles are supported" in str(e):
                        user_message = system_prompt + "\n\n" + example['instructions'][i]
                        chat = [
                            {
                                "role": "user",
                                "content": user_message
                            }
                        ]
                        if split == "train":
                            chat.append(
                                {
                                    "role": "assistant",
                                    "content": assistant_message
                                }
                            )
                        output_texts.append(tokenizer.apply_chat_template(chat, tokenize=False,
                                                                          add_generation_prompt=False if split == "train" else True))
                        label_texts.append(assistant_message)
                    else:
                        raise e
            if split == "train":
                return output_texts
            else:
                example["input_ids"] = tokenizer(output_texts,
                                                 padding=False, truncation=False,
                                                 add_special_tokens=False)["input_ids"]
                labels = example["input_ids"]
                example["labels"] = labels
                return example

        return formatting_chatml_func
    else:
        raise ValueError("Formatting style not supported")


def get_formatting_eval_func(annotated=False,
                             start_tag="[QUERY]", end_tag="[/QUERY]", few_shots=None):
    """Return formatting function based on the formatting style"""
    system_prompt = f"You are a helpful assistant. Your task is to provide a SPARQL query that answers the user's question. The query should be written between the tags {start_tag} and {end_tag}."
    if annotated:
        system_prompt = f"You are a WikiData chatbot. You will be given an instruction or a question. " \
                        "Convert the instruction or question into a SPARQL query. QIDs of entities and properties can be defined as text label.\n" \
                        f"Example: wdt:[property:instance of] instead of wdt:P31.\nThe query should be returned between the tags {start_tag} and {end_tag}."

    if few_shots:
        system_prompt += f"\n\n {few_shots}"

    def formatting_chatml_func(example):
        chat = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": example['instructions']
            }
        ]
        example["messages"] = chat
        return example

    return formatting_chatml_func
