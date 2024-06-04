# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, AutoPeftModelForCausalLM
from tokenizers import AddedToken
from tqdm import tqdm
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, HfArgumentParser, \
    Seq2SeqTrainingArguments, AutoTokenizer, GenerationConfig, DataCollatorForSeq2Seq
from transformers.utils import logging
from trl import setup_chat_format, is_xpu_available, is_npu_available

from data.wikidata import WikidataAPI
from dataset import load_data, get_formatting_func
from metrics import compute_metrics_with_metadata
from src.chat_template import FORMATS
from trainer import SFTTrainerGen, DataCollatorForCompletionOnlyLMEOS
from utils import MODELS_DIR

tqdm.pandas()

# os.environ["WANDB_MODE"] = "offline"
logger = logging.get_logger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    # General Arguments
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.2", metadata={"help": "the model name"})
    dataset_path: Optional[str] = field(
        default="PaDaS-Lab/Instruct-to-SPARQL", metadata={"help": "the dataset path"}
    )
    subset: Optional[str] = field(default="", metadata={"help": "the subset of the dataset"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    use_fast: Optional[bool] = field(default=False, metadata={"help": "whether to use fast tokenizer or not."})
    annotated_gen: Optional[bool] = field(default=False, metadata={"help": "whether the generation is annotated"})
    num_instructions: Optional[int] = field(default=1, metadata={"help": "the number of instructions to sample"})
    prompt_style: Optional[str] = field(default="chatml", metadata={"help": "The prompt style to use"})
    force_padding_side: Optional[bool] = field(default=False, metadata={"help": "The padding side to use"})
    add_generation_prompt: Optional[bool] = field(default=False,
                                                  metadata={"help": "Whether to add a generation prompt"})
    log_with: Optional[str] = field(default="wandb", metadata={"help": "use 'wandb' to log with wandb"})
    project_name: Optional[str] = field(default="sft-sparql", metadata={"help": "the project name"})
    group_by_length: Optional[bool] = field(default=False, metadata={"help": "whether to group by length"})
    packing: Optional[bool] = field(default=False, metadata={"help": "whether to use packing for SFTTrainer"})

    # Training Arguments
    per_device_train_batch_size: Optional[int] = field(default=8, metadata={"help": "the train batch size"})
    per_device_eval_batch_size: Optional[int] = field(default=32, metadata={"help": "the eval batch size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(default=False, metadata={"help": "Enable gradient checkpointing"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    eval_interval: Optional[int] = field(default=1, metadata={"help": "the number of eval steps"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})
    total_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=0.5, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    logging_steps: Optional[int] = field(default=2, metadata={"help": "Number of updates steps before two logs"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "Enable BF16 training"})
    fp16: Optional[bool] = field(default=False, metadata={"help": "Enable FP16 training"})
    auto_find_batch_size: Optional[bool] = field(default=False, metadata={"help": "Enable auto batch size finding"})

    # Optimizer and Scheduler
    optim: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    weight_decay: Optional[float] = field(default=1.0e-6, metadata={"help": "the weight decay"})
    adam_epsilon: Optional[float] = field(default=1e-8, metadata={"help": "the epsilon"})
    adam_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1"})
    adam_beta2: Optional[float] = field(default=0.95, metadata={"help": "the beta2"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the scheduler"})
    warmup_ratio: Optional[int] = field(default=0.1,
                                        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."})
    save_total_limit: Optional[int] = field(default=1, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    debugging: Optional[bool] = field(default=False, metadata={"help": "Enable debugging mode"})

    # Generation kwargs i.e max_new_tokens=1024, top_k=20, top_p=0.95, do_sample=True)
    max_new_tokens: Optional[int] = field(default=1024, metadata={"help": "the max number of new tokens"})
    top_k: Optional[int] = field(default=20, metadata={"help": "the top k"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "the temperature"})
    top_p: Optional[float] = field(default=0.7, metadata={"help": "the top p"})
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "the number of return sequences"})
    do_sample: Optional[bool] = field(default=False, metadata={"help": "whether to sample or not"})


parser = HfArgumentParser(ScriptArguments)

if __name__ == "__main__":
    accelerator = Accelerator()
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)

    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        torch_dtype=torch_dtype,
        token=script_args.use_auth_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name,
        use_fast=script_args.use_fast, trust_remote_code=script_args.trust_remote_code, )

    chat_template = None
    model_type = model.config.model_type if "Llama-3" not in script_args.model_name else "llama3"
    if tokenizer.chat_template is None and model_type not in FORMATS.keys():
        logger.warning(
            "Chat template not found in tokenizer. Setting up chat format with default chatml template, behavior may be unexpected.")
        model, tokenizer = setup_chat_format(model, tokenizer, format=script_args.prompt_style)
        chat_template = FORMATS[script_args.prompt_style]()
    elif model_type in FORMATS.keys():
        model, tokenizer = setup_chat_format(model, tokenizer, format=model_type)
        chat_template = FORMATS[model_type]()

    # If padding_side="right", set pad_token to unk_token
    if tokenizer.padding_side == "right":
        if script_args.force_padding_side:
            tokenizer.padding_side = "left"
        else:
            # change the pad_token
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else AddedToken("<|pad|>", lstrip=False,
                                                                                             rstrip=False)
    if model.config.model_type == "llama":
        tokenizer.padding_side = "right"  # OVERFLOW issues with left padding
    # Create GenerationConfig
    stop_strings = ["<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>", "<|end_of_text|>"]
    if script_args.do_sample:
        gen_kwargs = GenerationConfig(max_new_tokens=script_args.max_new_tokens, temperature=script_args.temperature,
                                      top_p=script_args.top_p, do_sample=script_args.do_sample,
                                      num_return_sequences=script_args.num_return_sequences,
                                      stop_strings=stop_strings, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id)
    else:
        gen_kwargs = GenerationConfig(max_new_tokens=script_args.max_new_tokens,
                                      num_return_sequences=script_args.num_return_sequences,
                                      stop_strings=stop_strings, eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id)

    # Step 2: Load the dataset
    dataset_name = script_args.dataset_path
    subset = script_args.subset
    annotated = script_args.annotated_gen
    model_name = f"{script_args.model_name.split('/')[-1]}-{dataset_name.split('/')[0]}-{subset}" if subset else f"{script_args.model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"
    checkpoint_dir = str(os.path.join(MODELS_DIR, script_args.project_name,
                                      f"{model_name}-sft-{script_args.lr_scheduler_type}-{script_args.prompt_style}-annotated-{annotated}"))
    with accelerator.main_process_first():
        dataset = load_data(dataset_name, subset=subset)
    # flatten the list of instructions field by creating new row for each instruction
    n_sample_instructions = script_args.num_instructions
    os.environ["WANDB_PROJECT"] = script_args.project_name


    def flatten_instructions(examples):
        new_examples = {k: [] for k in examples.keys()}
        limit = -1
        if script_args.debugging:
            limit = 5
        for i in range(len(examples["id"][:limit])):
            # sample a random instruction out of the list of instructions
            instructions = examples["instructions"][i]
            instructions = random.choices(instructions, k=n_sample_instructions)
            new_examples["id"].extend([examples["id"][i]] * n_sample_instructions)
            new_examples["instructions"].extend(instructions)
            new_examples["sparql_query"].extend([examples["sparql_query"][i]] * n_sample_instructions)
            new_examples["sparql_annotated"].extend([examples["sparql_annotated"][i]] * n_sample_instructions)
            new_examples["sparql_raw"].extend([examples["sparql_raw"][i]] * n_sample_instructions)
            new_examples["query_results"].extend([examples["query_results"][i]] * n_sample_instructions)
        return new_examples


    with accelerator.main_process_first():
        train_ds = dataset["train"].map(flatten_instructions, batched=True, num_proc=4, load_from_cache_file=False)
        validation_ds = dataset["validation"].map(flatten_instructions, batched=True, num_proc=4,
                                                  load_from_cache_file=False)
        test_ds = dataset["test"].map(flatten_instructions, batched=True, num_proc=4, load_from_cache_file=False)
        # Preprocess and Collators
        start_tag = "[QUERY]"
        end_tag = "[/QUERY]"
        train_formatting_func = get_formatting_func(tokenizer, formatting_style=script_args.prompt_style,
                                                    annotated=annotated,
                                                    split="train")
        train_ds = train_ds.map(lambda batch: {"text": train_formatting_func(batch)}, batched=True,
                                load_from_cache_file=False, num_proc=4)

        eval_formatting_func = get_formatting_func(tokenizer, formatting_style=script_args.prompt_style,
                                                   annotated=annotated,
                                                   split="eval")
        validation_ds = validation_ds.map(eval_formatting_func, batched=True,
                                          load_from_cache_file=False, num_proc=4)
        test_ds = test_ds.map(eval_formatting_func, batched=True,
                              load_from_cache_file=False, num_proc=4)

    if script_args.prompt_style == "chatml":
        response_template_with_context = chat_template.response_template
        response_template_assistant = chat_template.assistant
        # Get the response token ids for the assistant using the tokenizer and the response template with context
        response_template_assistant_ids = [tid for tid in
                                           tokenizer.encode(response_template_assistant, add_special_tokens=False) if
                                           tid in tokenizer.encode(response_template_with_context,
                                                                   add_special_tokens=False)]
    elif script_args.prompt_style == "prompts":
        response_template_with_context = "\n### Assistant:"
        response_template_assistant = "### Assistant:"
        response_template_assistant_ids = [tid for tid in
                                           tokenizer.encode(response_template_assistant, add_special_tokens=False) if
                                           tid in tokenizer.encode(response_template_with_context,
                                                                   add_special_tokens=False)]
    else:
        raise ValueError("Prompt style not supported")

    collator = DataCollatorForCompletionOnlyLMEOS(
        response_template=response_template_assistant_ids,
        tokenizer=tokenizer,
    )
    left_sided_tokenizer = None
    if tokenizer.padding_side == "right":
        left_sided_tokenizer = deepcopy(tokenizer)
        left_sided_tokenizer.padding_side = "left"
        eval_collator = DataCollatorForSeq2Seq(tokenizer=left_sided_tokenizer)
    else:
        eval_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    wikidata = WikidataAPI(start_tag=start_tag, end_tag=end_tag)
    compute_metrics = compute_metrics_with_metadata(validation_ds, tokenizer, wikidata, annotated=annotated,
                                                    report_to=script_args.log_with, accelerator=accelerator)

    # Step 3: Define the training arguments

    training_args = Seq2SeqTrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        optim=script_args.optim,
        learning_rate=script_args.learning_rate,
        weight_decay=script_args.weight_decay,
        adam_epsilon=script_args.adam_epsilon,
        adam_beta1=script_args.adam_beta1,
        adam_beta2=script_args.adam_beta2,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        logging_steps=script_args.logging_steps,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.total_steps,
        report_to=script_args.log_with,
        save_steps=script_args.save_steps,
        eval_strategy="epoch",
        logging_strategy="steps",
        eval_steps=script_args.eval_interval,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        predict_with_generate=True,
        auto_find_batch_size=script_args.auto_find_batch_size,
        generation_config=gen_kwargs,
        include_inputs_for_metrics=True,
        run_name=model_name,
        seed=script_args.seed,
        bf16=script_args.bf16,
        bf16_full_eval=script_args.bf16,
        fp16=script_args.fp16,
        fp16_full_eval=script_args.fp16,
        log_level="info",
    )

    # Step 4: Define the LoraConfig
    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Step 5: Define the Trainer
    dataset_kwargs = {"add_special_tokens": False}
    trainer = SFTTrainerGen(
        model=model,
        args=training_args,
        max_seq_length=script_args.seq_length,
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        dataset_text_field="text",
        packing=script_args.packing,
        peft_config=peft_config,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        left_sided_tokenizer=left_sided_tokenizer,
        data_collator=collator,
        eval_data_collator=eval_collator,
        dataset_kwargs=dataset_kwargs,
    )

    trainer.train()

    # Step 6: Save the model
    trainer.save_model(checkpoint_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

    # Free memory for merging weights and saving merged weights if we are using PEFT Lora
    if script_args.use_peft:
        del model
        if is_xpu_available():
            torch.xpu.empty_cache()
        elif is_npu_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(training_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)
