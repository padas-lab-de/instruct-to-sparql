import os

import evaluate
import numpy as np
import pandas as pd
import xgboost as xgb
from schema_matching.cal_column_similarity import create_similarity_matrix, this_directory, make_data_from
from schema_matching.train import test
from schema_matching.utils import table_column_filter, find_all_keys_values
from sklearn.metrics import f1_score

rouge = evaluate.load('rouge')
bleu = evaluate.load('bleu')

def make_df_from_json(data):
    """
    Adpated from source: https://github.com/fireindark707/Python-Schema-Matching
    Make csv file from json file.
    """
    # find key_values
    if isinstance(data, dict):
        key_values = find_all_keys_values(data, "")
    elif isinstance(data, list):
        key_values = find_all_keys_values({"TOPLEVEL": data}, "TOPLEVEL")
    else:
        raise ValueError('Your input JsonData is not a dictionary or list')

    key_values = {k.replace("TOPLEVEL.", ""): v for k, v in key_values.items() if len(v) > 1}

    df = pd.DataFrame({k: pd.Series(v) for k, v in key_values.items()})
    return df


def schema_matching(table1_df, table2_df, threshold=None, strategy="one-to-one", model_pth=None):
    """
    Adpated from source: https://github.com/fireindark707/Python-Schema-Matching
    Do Columns matching!
    """
    if model_pth is None:
        model_pth = str(this_directory / "model" / "2022-04-12-12-06-32")
    # filter columns
    table1_df = table_column_filter(table1_df)
    table2_df = table_column_filter(table2_df)

    # extract features
    features, _ = make_data_from(table1_df, table2_df, type="test")

    # load model and predict on features
    preds = []
    pred_labels_list = []
    for i in range(len(os.listdir(model_pth)) // 2):
        bst = xgb.Booster({'nthread': 4})  # init model
        bst.load_model(model_pth + "/" + str(i) + ".model")
        if threshold is not None:
            best_threshold = float(threshold)
        else:
            with open(model_pth + "/" + str(i) + ".threshold", 'r') as f:
                best_threshold = float(f.read())
        pred, pred_labels = test(bst, best_threshold, features, test_labels=np.ones(len(features)), type="inference")
        preds.append(pred)
        pred_labels_list.append(pred_labels)
        del bst

    df_pred, df_pred_labels, predicted_pairs = create_similarity_matrix(table1_df, table2_df, preds, pred_labels_list,
                                                                        strategy=strategy)
    return df_pred, df_pred_labels, predicted_pairs


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
        return 0.0 , 0.0 , 0.0
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

if __name__=="__main__":
    import time

    import pandas as pd
    from src.dataset import load_data
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
