import warnings

import numpy as np
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
from transformers import EvalPrediction
from transformers import logging
import random
from src.utils import significant

logger = logging.get_logger(__name__)


def compute_metrics_with_metadata(eval_dataset, tokenizer, wikidata, annotated=False, accelerator=None, report_to=None):
    def compute_metrics(eval_preds: EvalPrediction):
        from metrics_utils import bleu_score, rouge_score, qa_metrics, execution_metrics
        table = []
        prompts = eval_preds.inputs
        preds = eval_preds.predictions

        # Ignore any token with -100 label in processed texts
        outputs = np.where(preds != -100, preds, tokenizer.pad_token_id)
        prompts = np.where(prompts != -100, prompts, tokenizer.pad_token_id)
        # We then decode the predictions
        generated_texts = []
        for output in outputs:
            try:
                generated_texts.append(tokenizer.decode(output, skip_special_tokens=True))
            except Exception as e:
                warnings.warn(
                    f"Error in decoding output {output} with exception: {e} with token_pad_id: {tokenizer.pad_token_id}")
                logger.info(
                    f"Error in decoding output {output} with exception: {e} with token_pad_id: {tokenizer.pad_token_id}")
                generated_texts.append("")
        str_prompts = [tokenizer.decode(prompt, skip_special_tokens=True) for prompt in prompts]

        target_queries = list(eval_dataset["sparql_query"])

        if annotated:
            target_queries = list(eval_dataset["sparql_annotated"])

        if len(generated_texts) != len(target_queries):
            # Fix for https://github.com/huggingface/accelerate/issues/226
            warnings.warn(
                f"Number of processed queries {len(generated_texts)} does not match the number of target queries {len(target_queries)}.\n"
                f"Input Prompts: {len(str_prompts)}")
            generated_texts = generated_texts[:len(target_queries)]
            logger.info("Generated queries have been truncated to match the number of target queries.")

        # sample 5 random examples from the generated texts
        indices = random.sample(range(len(generated_texts)), 5)
        str_prompts = [str_prompts[i] for i in indices]
        generated_texts = [generated_texts[i] for i in indices]
        target_queries = [target_queries[i] for i in indices]
        columns = ["prompt", "model_output"]
        columns_data = [str_prompts, generated_texts]
        # Calculate metrics for the processed queries
        # step1: Need to extract and clean the processed queries

        generated_queries = [wikidata.extract_query(text) for text in generated_texts]

        columns.append("target_query")
        columns_data.append(target_queries)
        # step2: Calculate generation metrics
        metrics_dict = {}
        bleu_scores = bleu_score(target_queries, generated_queries)
        columns.append("bleu_score")
        columns_data.append(bleu_scores)
        metrics_dict.update({"bleu": np.mean(bleu_scores)})
        rouge_scores = rouge_score(target_queries, generated_queries)
        metrics_dict.update({k: np.mean(v) for k, v in rouge_scores.items()})
        qa = qa_metrics(target_queries, generated_queries)
        metrics_dict.update(qa)

        # step3: Calculate execution metrics
        # step3.1: Validate syntax of processed queries
        if annotated:
            # Replace the annotations of the entities and properties before executing the query
            processed_queries = [wikidata.replace_annotations(query) for query in generated_queries]
        else:
            processed_queries = generated_queries

        columns.append("extracted_query")
        columns_data.append(processed_queries)
        # step3.2: Execute the queries and calculate the execution metrics
        execution_results = list(eval_dataset["query_results"])
        execution_results = [execution_results[i] for i in indices]
        overlap_scores = []
        jaccard_scores = []
        dice_scores = []
        syntax_scores = []
        for i, query in tqdm(enumerate(processed_queries), desc="Executing queries", total=len(processed_queries)):
            try:
                response = wikidata.execute_sparql(query, timeout=60)
                syntax_scores.append(1.0)
                if response.success:
                    target_results = eval(execution_results[i])
                    overlap_score, jaccard_score, dice_score = execution_metrics(response.bindings, target_results)
                    overlap_scores.append(overlap_score)
                    jaccard_scores.append(jaccard_score)
                    dice_scores.append(dice_score)
                else:
                    overlap_scores.append(0.0)
                    jaccard_scores.append(0.0)
                    dice_scores.append(0.0)
            except SyntaxError as e:
                print(f"Error in getting target results: {e}")
                continue
            except QueryBadFormed as e:
                print("Syntax error in query")
                syntax_scores.append(0.0)
                overlap_scores.append(0.0)
                jaccard_scores.append(0.0)
                dice_scores.append(0.0)
                continue
            except Exception as e:
                print(f"Error in executing query: {e}")
                overlap_scores.append(0.0)
                jaccard_scores.append(0.0)
                dice_scores.append(0.0)
                continue

        metrics_dict.update({
            "overlap_score": sum(overlap_scores) / len(overlap_scores),
            "jaccard_score": sum(jaccard_scores) / len(jaccard_scores),
            "dice_score": sum(dice_scores) / len(dice_scores),
            "syntax_score": sum(syntax_scores) / len(syntax_scores)
        })

        # Add the syntax score to the columns
        columns.append("valid_syntax")
        columns_data.append(syntax_scores)

        table.append(list(zip(*columns_data)))
        rows = sum(list(map(list, zip(*table))), [])
        table_title = f"Evaluation Snippet results"
        rich_table = Table(*columns, title=table_title, show_lines=True)
        for ix in range(min(5, len(rows))):
            rich_table.add_row(*[str(significant(x)) for x in rows[ix]])
        Console().print(rich_table, markup=False)
        if report_to and report_to == "wandb":
            import wandb
            samples = wandb.Table(columns, rows)
            metrics_dict.update({"samples": samples})

        return metrics_dict

    def compute_metrics_empty(eval_preds: EvalPrediction):
        return {}

    if accelerator.is_main_process:
        return compute_metrics
    else:
        return compute_metrics_empty
