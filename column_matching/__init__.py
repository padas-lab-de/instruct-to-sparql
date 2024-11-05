import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from column_matching.relation_features import make_data_from
from column_matching.utils import table_column_filter, find_all_keys_values, test, create_similarity_matrix

this_directory = Path(__file__).parent


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

    key_values = {k.replace("TOPLEVEL.", ""): v for k, v in key_values.items() if len(v) >= 1}

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
