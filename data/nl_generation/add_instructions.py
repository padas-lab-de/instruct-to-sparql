import json
import os
import time
from pathlib import Path

import pandas as pd

base_dir = "/mnt/ext/Research/Data/"
output_dir = "/mnt/ext/Research/Dataset/"
os.makedirs(output_dir, exist_ok=True)


def load_data(json_path):
    """Load data from json file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def save_data(data, json_path):
    """Save data to json file"""
    with open(json_path, "w") as f:
        json.dump(data, f)


def add_instructions_to_data(split="all", dataset="final_fq18_without_limit"):
    """Add processed nl_generation to the data"""
    # Duplicates removal for the final version of the dataset
    rows_to_drop = [2673, 1527, 2782, 149, 155, 164, 1564, 306, 430, 438, 466, 514, 545, 876, 877, 853, 565, 865, 1503,
                    867, 868, 801, 805, 742, 765, 832, 858, 990, 1059, 1110, 1124, 1168, 1197, 1205, 1211, 1244, 1248,
                    1272, 1290, 1545, 1495, 1531, 1553, 1561, 1562, 1689, 1740, 2214, 1810, 1847, 1865, 1868, 1950,
                    1974, 2359, 2012, 2080, 2613, 2345, 2364, 2457, 2476, 2609, 2556, 2634, 2800, 2827]
    os.makedirs(os.path.join(output_dir, dataset), exist_ok=True)
    generated_instructions = load_data("../processed/sparql_instructions.json")
    generated_complexities = load_data("../processed/sparql_complexities.json")
    generated_instructions_df = pd.DataFrame(generated_instructions, columns=["id", "nl_generation"]).astype(
        {"id": int}).set_index("id")
    generated_instructions_df.drop(rows_to_drop, inplace=True)
    generated_instructions_df.reset_index(drop=True, inplace=True)
    generated_instructions_df.columns = ["nl_generation"]
    generated_complexities_df = pd.DataFrame(generated_complexities,
                                             columns=["id", "complexity", "complexity_description"]).astype(
        {"id": int}).set_index("id")
    generated_complexities_df.drop(rows_to_drop, inplace=True)
    generated_complexities_df.reset_index(drop=True, inplace=True)
    if split == "full":
        data = pd.read_parquet(os.path.join(base_dir, dataset, "generated_prompt-executed.parquet.gzip"),
                               engine="fastparquet")
        data.drop(["basic_prompt", "basic_num_tokens", "basic_result", "basic_is_skipped",
                   "basic_full_answer", "context", "description",
                   "basic_is_prompt_too_long", "templated_prompt", "templated_num_tokens", "templated_result",
                   "templated_is_skipped", "templated_is_prompt_too_long", "templated_full_answer"], axis=1,
                  inplace=True)
        data = data.join(generated_instructions_df)
        data = data.join(generated_complexities_df)
        # rearrange columns and rename them
        data.columns = ["sparql_raw", "sparql_annotated", "query_results", "sparql_query", "nl_generation", "complexity",
                        "complexity_description"]
        data["id"] = data.index
        data = data[["id", "nl_generation", "sparql_raw", "sparql_annotated", "sparql_query", "complexity",
                     "complexity_description", "query_results"]]
        data.to_parquet(
            os.path.join(output_dir, dataset, "text-to-sparql-full.parquet.gzip"), engine="fastparquet",
            compression="gzip", index=False)
    elif split == "train":
        data = pd.read_pickle(os.path.join(base_dir, dataset, "final_fq18-split_train.pkl"))
        data.drop(["basic_input", "templated_input"], axis=1, inplace=True)
        data = data.join(generated_instructions_df)
        data = data.join(generated_complexities_df)
        data.columns = ["sparql_raw", "sparql_annotated", "query_results", "sparql_query", "nl_generation", "complexity",
                        "complexity_description"]
        data["id"] = data.index
        data = data[["id", "nl_generation", "sparql_raw", "sparql_annotated", "sparql_query", "complexity",
                     "complexity_description", "query_results"]]
        data.to_parquet(
            os.path.join(output_dir, dataset, "text-to-sparql-train.parquet"), engine="pyarrow", index=False)
    elif split == "valid":
        data = pd.read_pickle(os.path.join(base_dir, dataset, "final_fq18-split_valid.pkl"))
        data.drop(["basic_input", "templated_input"], axis=1, inplace=True)
        data = data.join(generated_instructions_df)
        data = data.join(generated_complexities_df)
        data.columns = ["sparql_raw", "sparql_annotated", "query_results", "sparql_query", "nl_generation", "complexity",
                        "complexity_description"]
        data["id"] = data.index
        data = data[["id", "nl_generation", "sparql_raw", "sparql_annotated", "sparql_query", "complexity",
                     "complexity_description", "query_results"]]
        data.to_parquet(
            os.path.join(output_dir, dataset, "text-to-sparql-valid.parquet"), engine="pyarrow", index=False)
    elif split == "test":
        data = pd.read_pickle(os.path.join(base_dir, dataset, "final_fq18-split_test.pkl"))
        data.drop(["basic_input", "templated_input"], axis=1, inplace=True)
        data = data.join(generated_instructions_df)
        data = data.join(generated_complexities_df)
        data.columns = ["sparql_raw", "sparql_annotated", "query_results", "sparql_query", "nl_generation", "complexity",
                        "complexity_description"]
        data["id"] = data.index
        data = data[["id", "nl_generation", "sparql_raw", "sparql_annotated", "sparql_query", "complexity",
                     "complexity_description", "query_results"]]
        data.to_parquet(
            os.path.join(output_dir, dataset, "text-to-sparql-test.parquet"), engine="pyarrow", index=False)
    else:
        raise ValueError("Invalid split")


splits = ["full", "train", "valid", "test"]
# "train" ,"valid", "test"

for split in splits:
    add_instructions_to_data(split=split, dataset="final_fq18_without_limit")
    add_instructions_to_data(split=split, dataset="final_fq18_with_limit_10")
