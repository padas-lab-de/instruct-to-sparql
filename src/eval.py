import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
import tqdm
from data.wikidata import WikidataAPI
import pandas as pd
import torch
from transformers import HfArgumentParser
from metrics_utils import bleu_score, rouge_score, qa_metrics, execution_metrics
from src.dataset import load_data, get_formatting_eval_func
from src.utils import set_seed, get_generator_fn, MODELS_DIR

load_dotenv()

def flatten_instructions_for_eval(examples):
    k = 1
    new_examples = {k: [] for k in examples.keys()}
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
        response = wikidata.execute_sparql(processed_query, timeout=60)
        syntax_score = 1.0
        if response.success:
            overlap_score, jaccard_score, dice_score = execution_metrics(response.bindings, results)
        else:
            overlap_score = 0.0
            jaccard_score = 0.0
            dice_score = 0.0
    except QueryBadFormed as e:
        print("Syntax error in query")
        syntax_score = 0.0
        overlap_score = 0.0
        jaccard_score = 0.0
        dice_score = 0.0
    except Exception as e:
        print(f"Error in executing query: {e}")
        overlap_score = 0.0
        jaccard_score = 0.0
        dice_score = 0.0

    metrics_dict = {
        "bleu": bleu,
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
    model_name: Optional[str] = field(default="meta-llama/Meta-Llama-3-8B-Instruct",
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
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the learning rate scheduler type"})
    prompt_style: Optional[str] = field(default="chatml", metadata={"help": "the prompt style"})

    # Generation kwargs i.e max_new_tokens=1024, top_k=20, top_p=0.95, do_sample=True)
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    seed: Optional[int] = field(default=42, metadata={"help": "the seed"})
    max_new_tokens: Optional[int] = field(default=1024, metadata={"help": "the max number of new tokens"})
    top_k: Optional[int] = field(default=20, metadata={"help": "the top k"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "the temperature"})
    top_p: Optional[float] = field(default=0.7, metadata={"help": "the top p"})
    num_return_sequences: Optional[int] = field(default=1, metadata={"help": "the number of return sequences"})
    do_sample: Optional[bool] = field(default=True, metadata={"help": "whether to sample or not"})


parser = HfArgumentParser(ScriptArguments)

if __name__ == "__main__":
    script_args = parser.parse_args_into_dataclasses()[0]
    set_seed(script_args.seed)
    # Step 1: Load the model
    dtype = torch.float32 if "Llama-3" in script_args.model_name else torch.bfloat16
    model_kwargs = {"device_map": "auto",
                    "torch_dtype": dtype,
                    "trust_remote_code": True,
                    "token": script_args.use_auth_token}
    tokenizer_kwargs = {"use_fast": script_args.use_fast, "trust_remote_code": True}

    dataset_name = script_args.dataset_path
    subset = script_args.subset
    annotated = script_args.annotated_gen

    few_shot_eval = script_args.model_name == "llama3"

    if few_shot_eval:
        model_path = "llama3"
    else:
        model_name = f"{script_args.model_name.split('/')[-1]}-{dataset_name.split('/')[0]}-{subset}" if subset else f"{script_args.model_name.split('/')[-1]}-{dataset_name.split('/')[-1]}"
        checkpoint_dir = str(os.path.join(MODELS_DIR, script_args.project_name,
                                          f"{model_name}-sft-{script_args.lr_scheduler_type}-{script_args.prompt_style}-annotated-{annotated}"))
        model_path = os.path.join(checkpoint_dir)

    # Create GenerationConfig
    stop_strings = ["<|endoftext|>", "<|end|>", "</s>", "<|eot_id|>", "<|end_of_text|>"]
    if script_args.do_sample:
        gen_kwargs = dict(max_new_tokens=script_args.max_new_tokens, temperature=script_args.temperature,
                          top_p=script_args.top_p, do_sample=script_args.do_sample,
                          num_return_sequences=script_args.num_return_sequences,
                          stop_strings=stop_strings)
    else:
        gen_kwargs = dict(max_new_tokens=script_args.max_new_tokens,
                          num_return_sequences=script_args.num_return_sequences,
                          stop_strings=stop_strings)
    # Load the dataset
    dataset = load_data(dataset_name, subset=subset)
    test_ds = dataset["test"].map(flatten_instructions_for_eval, batched=True, num_proc=4, load_from_cache_file=False)
    if script_args.num_samples != -1:
        test_ds = test_ds.select(range(script_args.num_samples))
    target_key = "sparql_query"
    if script_args.annotated_gen:
        target_key = "sparql_annotated"

    # Few-shot examples for the llama3 model
    start_tag = "[QUERY]"
    end_tag = "[/QUERY]"
    few_shots = None
    if few_shot_eval:
        # select 2 short examples from the train split
        num_shots = 2
        few_shot_examples = dataset["train"].filter(lambda x: len(x[target_key].split(" ")) < 150).select(range(num_shots))
        few_shots = f"Examples:\n"
        for i in range(len(few_shot_examples[target_key])):
            few_shots += f"- User: {few_shot_examples['instructions'][i][0]}\nAnswer: Here is the SPARQL query: \n{start_tag}\n{few_shot_examples[target_key][i]}\n{end_tag}\n"

        gen_kwargs = {"max_tokens": script_args.max_new_tokens, "temperature": script_args.temperature,
                      "top_p": script_args.top_p, "n": script_args.num_return_sequences}
    formatting_fn = get_formatting_eval_func(annotated=script_args.annotated_gen, few_shots=few_shots)
    test_dataset = test_ds.map(formatting_fn, load_from_cache_file=False)

    # Load the model
    generator_fn = get_generator_fn(model_path, model_kwargs, tokenizer_kwargs, huggingface=not few_shot_eval)

    if not few_shot_eval:
        gen_kwargs.update({"return_full_text": False, "batch_size": 64})

    # Step 2: Generate the SPARQL queries
    wikidata = WikidataAPI(start_tag="[QUERY]", end_tag="[/QUERY]")
    table = defaultdict(list)
    for i, response in tqdm.tqdm(enumerate(generator_fn(messages=test_dataset["messages"], **gen_kwargs)), total=len(test_dataset["messages"]), desc="Evaluating Generations"):
        message = test_dataset["messages"][i]
        target_query = test_dataset[target_key][i]
        generated_output = response[0]["generated_text"] if not few_shot_eval else response
        table["message"].append(message)
        table["target_query"].append(target_query)
        table["complexity"].append(test_dataset["complexity"][i])
        table["generated_query"].append(generated_output)
    if few_shot_eval:
        model_path = f"/data/{model_path}/annotated-{annotated}"
        os.makedirs(model_path, exist_ok=True)
    generations = pd.DataFrame(table)
    generations.to_csv(f"{model_path}/generated_queries.csv", index=False)
    for i in range(len(table["message"])):  # for each generated query
        generated_output = table["generated_query"][i]
        target_query = table["target_query"][i]
        try:
            # step3: Calculate generation metrics
            results = eval(test_dataset["query_results"][i])
            metrics = compute_metrics_eval(generated_output, target_query, results, wikidata=wikidata,
                                           annotated=annotated)
            table["bleu"].append(metrics["bleu"])
            table["f1"].append(metrics["f1"])
            table["em"].append(metrics["exact_match"])
            table["overlap_score"].append(metrics["overlap_score"])
            table["jaccard_score"].append(metrics["jaccard_score"])
            table["dice_score"].append(metrics["dice_score"])
            table["syntax_score"].append(metrics["syntax_score"])
            table["rouge1"].append(metrics["rouge1"][0])
            table["rouge2"].append(metrics["rouge2"][0])
            table["rougeL"].append(metrics["rougeL"][0])
            table["rougeLsum"].append(metrics["rougeLsum"][0])
        except Exception as e:
            print(f"Error in computing metrics: {e}")
            table["bleu"].append(0.0)
            table["f1"].append(0.0)
            table["em"].append(0.0)
            table["overlap_score"].append(0.0)
            table["jaccard_score"].append(0.0)
            table["dice_score"].append(0.0)
            table["syntax_score"].append(0.0)
            table["rouge1"].append(0.0)
            table["rouge2"].append(0.0)
            table["rougeL"].append(0.0)
            table["rougeLsum"].append(0.0)

    # Step 3: Save the results
    df_results = pd.DataFrame(table)
    df_results.to_csv(f"{model_path}/test_results.csv", index=False)
    # Step 4: Print the aggregated metrics for the test set
    print(f"#### Aggregated Metrics for {model_path} on {dataset_name} with annotation={annotated} ####")
    print(f"BLEU: {df_results['bleu'].mean()}")
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
