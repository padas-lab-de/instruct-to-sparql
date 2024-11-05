import logging

import evaluate
import numpy as np
from sklearn.metrics import f1_score
from codebleu import calc_codebleu
from rdflib.plugins.sparql.parser import parseQuery
from column_matching import make_df_from_json, schema_matching

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')


def overlap(X, Y):
    """ Compute the overlap between two sets of values """
    return len(set(X) & set(Y)) / min(len(set(X)), len(set(Y)))


def jaccard_index(X, Y):
    """ Compute the Jaccard index between two sets of values """
    return len(set(X) & set(Y)) / len(set(X) | set(Y))


def dice_coefficient(X, Y):
    """ Compute the Dice coefficient between two sets of values """
    return 2 * len(set(X) & set(Y)) / (len(set(X)) + len(set(Y)))


def bleu_score(references, generated_texts):
    """ Compute the BLEU score between a list of references and a list of processed texts """
    scores = []
    for reference, generated_text in zip(references, generated_texts):
        if len(generated_text) == 0:
            scores.append(0)
            continue

        results = bleu.compute(references=[[reference]], predictions=[generated_text])
        scores.append(results.get("bleu", 0.0))
    return scores


def codebleu_score(references, generated_texts):
    """ Compute the CodeBLEU score between a list of references and a list of processed texts """
    scores = []
    for reference, generated_text in zip(references, generated_texts):
        if len(generated_text) == 0:
            scores.append(0)
            continue
        try:
            parsed_ref = str(parseQuery(reference))
            parsed_gen = str(parseQuery(generated_text))
            results = calc_codebleu([[parsed_ref]], [parsed_gen], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                                    tokenizer=None)
            scores.append(results["codebleu"])
        except Exception as e:
            logging.warning(
                "WARNING: Parsing the query failed and codebleu reverts to python ast parsing. Please consider ignoring this score. Error: %s",
                e)
            results = calc_codebleu([[reference]], [generated_text], lang="python", weights=(0.25, 0.25, 0.25, 0.25),
                                    tokenizer=None)
            scores.append(results["codebleu"])

    return scores


def rouge_score(references, generated_texts):
    """ Compute the ROUGE score between a list of references and a list of processed texts """
    return rouge.compute(predictions=generated_texts, references=references, use_aggregator=False)


def qa_metrics(labels, preds):
    """ Compute the exact match score between a list of labels and a list of predictions """
    em = np.mean([label == pred for label, pred in zip(labels, preds)])
    f1 = f1_score(labels, preds, average="macro")

    return {"exact_match": em, "f1": f1}


def execution_metrics(results, target_results):
    """ Compute the execution metrics between the results and the target results """
    if not results:
        return 0.0, 0.0, 0.0
    else:
        results = make_df_from_json(results)
        target_results = make_df_from_json(target_results)
        # schema matching of columns
        df_pred, df_pred_labels, predicted_pairs = schema_matching(results, target_results)

        # calculate metrics for each column pair values
        overlap_scores = []
        jaccard_scores = []
        dice_scores = []
        for col, target_col, score in predicted_pairs:
            values = results[col].values
            target_values = target_results[target_col].values
            overlap_scores.append(overlap(values, target_values))
            jaccard_scores.append(jaccard_index(values, target_values))
            dice_scores.append(dice_coefficient(values, target_values))

        overlap_score = np.mean(overlap_scores)
        jaccard_score = np.mean(jaccard_scores)
        dice_score = np.mean(dice_scores)
        return overlap_score, jaccard_score, dice_score


if __name__ == "__main__":
    import time
    from src.data.dataset import load_data
    from data.wikidata import WikidataAPI

    wikidata = WikidataAPI()
    dataset = load_data("PaDaS-Lab/Instruct-to-SPARQL")
    eval_dataset = dataset["validation"]

    # query = eval_dataset["sparql_query"][0]
    # results = wikidata.execute_sparql(query)
    # valid_syntax = wikidata.validate_syntax(query)
    table1 = eval(eval_dataset["query_results"][0])
    table2 = eval(eval_dataset["query_results"][0])
    start = time.time()
    metrics = execution_metrics(table1, table2)
    print("schema_matching|Time taken:", time.time() - start)
