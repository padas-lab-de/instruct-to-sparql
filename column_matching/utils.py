from collections import defaultdict
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd

def test(bst,best_threshold, test_features, test_labels, type="evaluation"):
    dtest = xgb.DMatrix(test_features, label=test_labels)
    pred = bst.predict(dtest)
    if type == "inference":
        pred_labels = np.where(pred > best_threshold, 1, 0)
        return pred,pred_labels
    # compute precision, recall, and F1 score
    pred_labels = np.where(pred > best_threshold, 1, 0)
    precision = precision_score(test_labels, pred_labels,average="binary",pos_label=1)
    recall = recall_score(test_labels, pred_labels,average="binary",pos_label=1)
    f1 = f1_score(test_labels, pred_labels,average="binary",pos_label=1)
    c_matrix = confusion_matrix(test_labels, pred_labels)
    return precision, recall, f1, c_matrix

def find_all_keys_values(json_data,parent_key):
    """
    Find all keys that don't have list or dictionary values and their values. Key should be saved with its parent key like "parent-key.key".
    """
    key_values = defaultdict(list)
    for key, value in json_data.items():
        if isinstance(value, dict):
            child_key_values = find_all_keys_values(value,key)
            for child_key, child_value in child_key_values.items():
                key_values[child_key].extend(child_value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    child_key_values = find_all_keys_values(item,key)
                    for child_key, child_value in child_key_values.items():
                        key_values[child_key].extend(child_value)
                else:
                    key_values[parent_key+"."+key].append(item)
        else:
            key_values[parent_key+"."+key].append(value)
    return key_values

def table_column_filter(table_df):
    """
    Filter columns that have zero instances or all columns are "--"
    """
    original_columns = table_df.columns
    for column in table_df.columns:
        column_data = [d for d in list(table_df[column]) if d == d and d != "--"]
        if len(column_data) < 1:
            table_df = table_df.drop(column, axis=1)
            continue
        if "Unnamed:" in column:
            table_df = table_df.drop(column, axis=1)
            continue
    remove_columns = list(set(original_columns) - set(table_df.columns))
    if len(remove_columns) > 0:
        print("Removed columns:", remove_columns)
    return table_df


def create_similarity_matrix(table1_df, table2_df, preds, pred_labels_list, strategy="many-to-many"):
    """
    Create a similarity matrix from the prediction
    """
    predicted_pairs = []
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)
    pred_labels_list = np.array(pred_labels_list)
    pred_labels = np.mean(pred_labels_list, axis=0)
    pred_labels = np.where(pred_labels > 0.5, 1, 0)
    # read column names
    df1_cols = table1_df.columns
    df2_cols = table2_df.columns
    # create similarity matrix for pred values
    preds_matrix = np.array(preds).reshape(len(df1_cols), len(df2_cols))
    # create similarity matrix for pred labels
    if strategy == "many-to-many":
        pred_labels_matrix = np.array(pred_labels).reshape(len(df1_cols), len(df2_cols))
    else:
        pred_labels_matrix = np.zeros((len(df1_cols), len(df2_cols)))
        for i in range(len(df1_cols)):
            for j in range(len(df2_cols)):
                if pred_labels[i * len(df2_cols) + j] == 1:
                    if strategy == "one-to-one":
                        max_row = max(preds_matrix[i, :])
                        max_col = max(preds_matrix[:, j])
                        if preds_matrix[i, j] == max_row and preds_matrix[i, j] == max_col:
                            pred_labels_matrix[i, j] = 1
                    elif strategy == "one-to-many":
                        max_row = max(preds_matrix[i, :])
                        if preds_matrix[i, j] == max_row:
                            pred_labels_matrix[i, j] = 1
    df_pred = pd.DataFrame(preds_matrix, columns=df2_cols, index=df1_cols)
    df_pred_labels = pd.DataFrame(pred_labels_matrix, columns=df2_cols, index=df1_cols)
    for i in range(len(df_pred_labels)):
        for j in range(len(df_pred_labels.iloc[i])):
            if df_pred_labels.iloc[i, j] == 1:
                predicted_pairs.append((df_pred.index[i], df_pred.columns[j], df_pred.iloc[i, j]))
    return df_pred, df_pred_labels, predicted_pairs