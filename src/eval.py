import gc
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import json
import pandas as pd
import torch
import tqdm
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError
from _socket import timeout
from dotenv import load_dotenv
from requests import Timeout
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig
from vllm.distributed import destroy_model_parallel

from chat_template import FORMATS
from data.wikidata import WikidataAPI
from dataset import load_data, get_formatting_eval_func
from metrics_utils import bleu_score, rouge_score, qa_metrics, execution_metrics, codebleu_score
from utils import set_seed, get_generator_fn, jdump

# if vllm is available, import it
try:
    from vllm import LLM, SamplingParams
except ImportError:
    raise ImportError("Please install the vllm package to run this script")
load_dotenv()


def flatten_instructions_for_eval(examples):
    k = 1
    new_examples = {key: [] for key in examples.keys()}
    for i in range(len(examples["id"])):
        # sample a random instruction out of the list of instructions
        instructions = examples["instructions"][i]
        instructions = instructions[:k]
        new_examples["id"].extend([examples["id"][i]] * k)
        new_examples["instructions"].extend(instructions)
        new_examples["sparql_query"].extend([examples["sparql_query"][i]] * k)
        new_examples["sparql_annotated"].extend([examples["sparql_annotated"][i]] * k)
        new_examples["sparql_raw"].extend([examples["sparql_raw"][i]] * k)
        new_examples["complexity"].extend([examples["complexity"][i]] * k)
        new_examples["complexity_description"].extend([examples["complexity_description"][i]] * k)
        new_examples["query_results"].extend([examples["query_results"][i]] * k)
    return new_examples


def compute_metrics_eval(generated, target, results, wikidata, annotated=False):
    query = wikidata.extract_query(generated)

    # Calculate generation metrics
    bleu = bleu_score([target], [query])[0]
    codebleu = codebleu_score([target], [query])[0]
    rouge = rouge_score([target], [query])
    qa = qa_metrics([target], [query])
    # Calculate execution metrics
    if annotated:
        # Replace the annotations of the entities and properties before executing the query
        processed_query = wikidata.replace_annotations(query)
    else:
        processed_query = query
    syntax_score = 1.0
    try:
        gen_results = wikidata.execute_sparql(processed_query, timeout=60)
        syntax_score = 1.0
        if gen_results.success:
            overlap_score, jaccard_score, dice_score = execution_metrics(gen_results.bindings, results)
        else:
            overlap_score = 0.0
            jaccard_score = 0.0
            dice_score = 0.0
    except QueryBadFormed as e:
        syntax_score = 0.0
        overlap_score = 0.0
        jaccard_score = 0.0
        dice_score = 0.0
    except EndPointInternalError as e:
        if not results:
            overlap_score = 1.0
            jaccard_score = 1.0
            dice_score = 1.0
        else:
            overlap_score = 0.0
            jaccard_score = 0.0
            dice_score = 0.0
    except timeout or Timeout as e:
        overlap_score = 0.0
        jaccard_score = 0.0
        dice_score = 0.0
    except Exception as e:
        raise e

    metrics_dict = {
        "bleu": bleu,
        "codebleu": codebleu,
        "exact_match": qa["exact_match"],
        "f1": qa["f1"],
        "overlap_score": overlap_score,
        "jaccard_score": jaccard_score,
        "dice_score": dice_score,
        "syntax_score": syntax_score
    }
    metrics_dict.update({k: v for k, v in rouge.items()})

    return metrics_dict


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    # General Arguments
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-Instruct-v0.3",
                                      metadata={"help": "the model name"})
    dataset_path: Optional[str] = field(
        default="PaDaS-Lab/Instruct-to-SPARQL", metadata={"help": "the dataset path"}
    )
    subset: Optional[str] = field(default="default", metadata={"help": "the subset of the dataset"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    use_fast: Optional[bool] = field(default=False, metadata={"help": "whether to use fast tokenizer or not."})
    annotated_gen: Optional[bool] = field(default=False, metadata={"help": "whether the generation is annotated"})
    num_samples: Optional[int] = field(default=-1, metadata={"help": "the number of samples to test"})
    use_auth_token: Optional[bool] = field(default=True, metadata={"help": "Use HF auth token to access the model"})
    project_name: Optional[str] = field(default="sft-sparql", metadata={"help": "the project name"})
    prompt_style: Optional[str] = field(default="chatml", metadata={"help": "the prompt style"})

    # Generation kwargs i.e max_new_tokens=1024, top_k=20, top_p=0.95, do_sample=True)
    n_shots: Optional[int] = field(default=3, metadata={"help": "How many shot examples to use for few shot eval"})
    agent_id: Optional[int] = field(default=0, metadata={"help": "the agent id"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    max_new_tokens: Optional[int] = field(default=1024, metadata={"help": "the max number of new tokens"})
    top_k: Optional[int] = field(default=20, metadata={"help": "the top k"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "the temperature"})
    top_p: Optional[float] = field(default=0.7, metadata={"help": "the top p"})
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "the number of return sequences"})
    do_sample: Optional[bool] = field(default=True, metadata={"help": "whether to sample or not"})


parser = HfArgumentParser(ScriptArguments)

few_shot_eval_type = {
    "gpt-4": True,
    "gpt-4o": True,
    "gpt-4o-mini": True,
    "gpt-3.5-turbo-0125": True,
    "gpt-3.5-turbo-1106": True,
    "llama3.1": True,
    "qwen2.5": True,
    "meta-llama/Meta-Llama-3-8B-Instruct": True,
    "mistralai/Mistral-7B-Instruct-v0.3": True,
    "Qwen/Qwen2.5-0.5B-Instruct": True,
    "PaDaS-Lab/Llama3-8B-SPARQL": False,
    "PaDaS-Lab/Mistral-7B-v0.3-SPARQL": False,
    "PaDaS-Lab/Llama3-8B-SPARQL-annotated": False,
    "PaDaS-Lab/Mistral-7B-v0.3-SPARQL-annotated": False,
    "/data/llms/sft-sparql/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-False": False,
    "/data/llms/sft-sparql-annotated/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-True": False
}

vllm_eval_type = {
    "gpt-4": False,
    "gpt-4o": False,
    "gpt-4o-mini": False,
    "gpt-3.5-turbo-0125": False,
    "gpt-3.5-turbo-1106": False,
    "llama3.1": False,
    "qwen2.5": False,
    "meta-llama/Meta-Llama-3-8B-Instruct": True,
    "mistralai/Mistral-7B-Instruct-v0.3": True,
    "Qwen/Qwen2.5-0.5B-Instruct": True,
    "PaDaS-Lab/Llama3-8B-SPARQL": True,
    "PaDaS-Lab/Mistral-7B-v0.3-SPARQL": True,
    "PaDaS-Lab/Llama3-8B-SPARQL-annotated": True,
    "PaDaS-Lab/Mistral-7B-v0.3-SPARQL-annotated": True,
    "/data/llms/sft-sparql/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-False": True,
    "/data/llms/sft-sparql-annotated/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-True": True
}

if __name__ == "__main__":
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)
    # Step 1: Load the model
    tokenizer_kwargs = {"use_fast": script_args.use_fast, "trust_remote_code": True}

    dataset_name = script_args.dataset_path
    subset = script_args.subset
    annotated = script_args.annotated_gen

    few_shot_eval = few_shot_eval_type[script_args.model_name]
    vllm_eval = vllm_eval_type[script_args.model_name]
    # Create GenerationConfig
    stop_strings = ["<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>", "<|end_of_text|>"]
    if script_args.do_sample:
        sampling_params = SamplingParams(max_tokens=script_args.max_new_tokens, temperature=script_args.temperature,
                                         top_p=script_args.top_p,
                                         n=script_args.num_return_sequences,
                                         stop=stop_strings)
    else:
        sampling_params = SamplingParams(max_tokens=script_args.max_new_tokens, n=script_args.num_return_sequences,
                                         stop=stop_strings)
    # Load the dataset
    dataset = load_data(dataset_name, subset=subset)
    test_ds = dataset["test"].map(flatten_instructions_for_eval, batched=True, num_proc=4, load_from_cache_file=False)
    if script_args.num_samples != -1:
        test_ds = test_ds.select(range(script_args.num_samples))
    target_key = "sparql_query"
    if script_args.annotated_gen:
        target_key = "sparql_annotated"

    # Few-shot examples for few shot evaluation
    start_tag = "[QUERY]"
    end_tag = "[/QUERY]"
    few_shots = None
    gen_kwargs = None
    if few_shot_eval:
        # select k-shot examples from the top 10 similar queries
        num_shots = script_args.n_shots
        few_shot_examples = []
        if num_shots > 0:
            filtered_train = dataset["train"].filter(lambda x: len(x["sparql_query"].split(" ")) < 300)
            with open("data/few_shots.json", "r") as f:
                few_shot_indices = json.load(f)
            
            for i in range(len(test_ds)):
                few_shots = f"Examples:\n"
                indices = few_shot_indices[str(i)]
                for j in range(num_shots):
                    few_shots += f"- User: {filtered_train[indices[j]]['instructions'][0]}\nAnswer: Here is the SPARQL query: \n{start_tag}\n{filtered_train[indices[j]][target_key]}\n{end_tag}\n"
                few_shot_examples.append(few_shots)
        else:
            # For zero-shot, we'll add an empty string for each example
            few_shot_examples = [""] * len(test_ds)

        test_ds = test_ds.add_column("few_shots", few_shot_examples)
        gen_kwargs = {"max_tokens": script_args.max_new_tokens, "temperature": script_args.temperature,
                      "top_p": script_args.top_p, "n": script_args.num_return_sequences}
    if vllm_eval:
        config = AutoConfig.from_pretrained(script_args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, **tokenizer_kwargs)
        # update chat template
        chat_format = FORMATS.get(config.model_type, FORMATS["chatml"])()
        if tokenizer.chat_template != chat_format.chat_template:
            tokenizer.chat_template = chat_format.chat_template
    else:
        tokenizer = None
    formatting_fn = get_formatting_eval_func(tokenizer=tokenizer, annotated=script_args.annotated_gen,
                                             few_shot_eval=few_shot_eval)
    test_dataset = test_ds.map(formatting_fn, load_from_cache_file=False)

    # Step 2: Generate the SPARQL queries
    output_path = f"/data/results/sparql-15-08/{script_args.n_shots}_shots/{script_args.model_name.split('/')[-1]}/annotated-{annotated}"
    os.makedirs(output_path, exist_ok=True)
    wikidata = WikidataAPI(start_tag="[QUERY]", end_tag="[/QUERY]", agent_id=script_args.agent_id)
    if not os.path.exists(f"{output_path}/generated_queries.csv"):
        if not vllm_eval:
            eval_gen = get_generator_fn(script_args.model_name, output_path)
            generations = eval_gen(messages=test_dataset["messages"], **gen_kwargs)
        else:
            tokenizer_mode = "auto"
            if not script_args.use_fast:
                tokenizer_mode = "slow"
            eval_gen = LLM(model=script_args.model_name, trust_remote_code=True, seed=script_args.seed,
                           tensor_parallel_size=2, max_num_seqs=script_args.batch_size, tokenizer_mode=tokenizer_mode, )
            # Generate the SPARQL queries in batches
            generations = eval_gen.generate(test_dataset["prompt"], sampling_params=sampling_params)
            # Delete the llm object and free the memory
            destroy_model_parallel()
            del eval_gen
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            print("Successfully delete the llm pipeline and free the GPU memory!")
        table = defaultdict(list)
        for i, response in tqdm.tqdm(enumerate(generations),
                                     total=len(generations), desc="Generating Queries"):
            prompt = response.prompt if vllm_eval else test_dataset["messages"][i]
            target_query = test_dataset[target_key][i]
            generated_output = response.outputs[0].text if vllm_eval else response
            table["message"].append(prompt)
            table["user_instruction"].append(test_dataset["instructions"][i])
            table["target_query"].append(target_query)
            table["complexity"].append(test_dataset["complexity"][i])
            table["generated_query"].append(generated_output)

        generations = pd.DataFrame(table)
        generations.to_csv(f"{output_path}/generated_queries.csv", index=False)
    else:
        generations = pd.read_csv(f"{output_path}/generated_queries.csv")
    results_table = defaultdict(list)
    n_errors = 0
    errors_dict = defaultdict(list)
    for i in tqdm.tqdm(range(generations.shape[0]), desc="Evaluating Generations"):  # for each generated query
        generated_output = generations["generated_query"].loc[i]
        target_query = generations["target_query"].loc[i]
        try:
            # step3: Calculate generation metrics
            # results = eval(test_dataset["query_results"][i])
            target_execution = wikidata.execute_sparql(target_query, timeout=60)
            results = target_execution.bindings
            metrics = compute_metrics_eval(generated_output, target_query, results,
                                           wikidata=wikidata,
                                           annotated=annotated)
            results_table["bleu"].append(metrics["bleu"])
            results_table["codebleuv2"].append(metrics["codebleu"])
            results_table["f1"].append(metrics["f1"])
            results_table["em"].append(metrics["exact_match"])
            results_table["overlap_score"].append(metrics["overlap_score"])
            results_table["jaccard_score"].append(metrics["jaccard_score"])
            results_table["dice_score"].append(metrics["dice_score"])
            results_table["syntax_score"].append(metrics["syntax_score"])
            results_table["rouge1"].append(metrics["rouge1"][0])
            results_table["rouge2"].append(metrics["rouge2"][0])
            results_table["rougeL"].append(metrics["rougeL"][0])
            results_table["rougeLsum"].append(metrics["rougeLsum"][0])
        except Exception as e:
            n_errors += 1
            errors_dict[f"{str(e)}"].append(i)
            continue

    # Step 3: Save the results and errors
    jdump(errors_dict, f"{output_path}/errors.json")

    df_results = pd.DataFrame(results_table)
    df_results.to_csv(f"{output_path}/test_results.csv", index=False)
    # Step 4: Print the aggregated metrics for the test set
    print(f"#### Number of Errors: {n_errors} ####")
    print(f"#### Aggregated Metrics for {script_args.model_name} on {dataset_name} with annotation={annotated} ####")
    print(f"BLEU: {df_results['bleu'].mean()}")
    print(f"Codebleu: {df_results['codebleuv2'].mean()}")
    print(f"EM: {df_results['em'].mean()}")
    print(f"F1: {df_results['f1'].mean()}")
    print(f"Overlap Score: {df_results['overlap_score'].mean()}")
    print(f"Jaccard Score: {df_results['jaccard_score'].mean()}")
    print(f"Dice Score: {df_results['dice_score'].mean()}")
    print(f"Syntax Score: {df_results['syntax_score'].mean()}")
    print(f"Rouge1: {df_results['rouge1'].mean()}")
    print(f"Rouge2: {df_results['rouge2'].mean()}")
    print(f"RougeL: {df_results['rougeL'].mean()}")
    print(f"RougeLsum: {df_results['rougeLsum'].mean()}")
