import gc
import os
import pandas as pd
import datasets
from datasets import DatasetDict, load_dataset
from datasets.arrow_dataset import Dataset
from huggingface_hub import login, HfApi

base_dir = "/mnt/ext/Research/Dataset/"
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN, write_permission=True)


def check_or_create_dataset_repo(token, dataset_name, organization=None):
    """
    Check if a dataset repo exists on HuggingFace, and if not, creates it.

    Args:
    - token (str): Your HuggingFace authentication token.
    - dataset_name (str): The name of the dataset repo.
    - organization (str, optional): The name of the organization under which the dataset should be created.
                                    If not specified, the dataset will be created under the authenticated user's account.

    Returns:
    - str: A message indicating the status of the repo.
    """

    api = HfApi()

    # Check if the dataset exists
    repo_url = f"{organization}/{dataset_name}" if organization else dataset_name

    # If the dataset doesn't exist, create a new dataset repo
    api.create_repo(token=token, repo_id=repo_url, repo_type="dataset", private=True, exist_ok=True)
    return repo_url


def create_hf_dataset(dataset_path, local_name="final_fq18_without_limit", hub_name="Instruct-to-SPARQL"):
    repo_id = check_or_create_dataset_repo(HF_TOKEN, hub_name, organization="PaDaS-Lab")
    features = datasets.Features(
        {
            "id": datasets.Value("int16"),
            "instructions": datasets.Sequence(datasets.Value("string")),
            "sparql_raw": datasets.Value("string"),
            "sparql_annotated": datasets.Value("string"),
            "sparql_query": datasets.Value("string"),
            "complexity": datasets.Value("string"),
            "complexity_description": datasets.Value("string"),
            "query_results": datasets.Value("string"),
        }
    )
    description = datasets.DatasetInfo(
        description=f"""Text-to-SPARQL Dataset which consists of pairs of Natural language instructions and SPARQL queries.\
                    The dataset is created by crawling the Wikipedia pages for tutorials and examples. The dataset has a total of 2.8k examples.\
                    The dataset is split into train, validation and test sets.""",
        features=features,
    )

    dataset = DatasetDict()
    all_df = pd.read_parquet(os.path.join(dataset_path, local_name, "text-to-sparql-full.parquet.gzip"),
                             engine="fastparquet")
    all_df.reset_index(drop=True, inplace=True)
    all_df = all_df.astype({"id": "int16", "sparql_raw": "string", "sparql_annotated": "string",
                            "sparql_query": "string", "query_results": "string", "complexity": "string",
                            "complexity_description": "string"})
    dataset["full"] = Dataset.from_pandas(
        all_df,
        features=features, info=description)
    del all_df
    gc.collect()
    train_df = pd.read_parquet(os.path.join(dataset_path, local_name, "text-to-sparql-train.parquet"), engine="pyarrow")
    train_df.reset_index(drop=True, inplace=True)
    train_df = train_df.astype({"id": "int16", "sparql_raw": "string", "sparql_annotated": "string",
                                "sparql_query": "string", "query_results": "string", "complexity": "string",
                                "complexity_description": "string"})
    dataset["train"] = Dataset.from_pandas(
        train_df,
        features=features, info=description)
    del train_df
    gc.collect()
    valid_df = pd.read_parquet(os.path.join(dataset_path, local_name, "text-to-sparql-valid.parquet"), engine="pyarrow")
    valid_df.reset_index(drop=True, inplace=True)
    valid_df = valid_df.astype({"id": "int16", "sparql_raw": "string", "sparql_annotated": "string",
                                "sparql_query": "string", "query_results": "string",
                                "complexity": "string", "complexity_description": "string"})
    dataset["valid"] = Dataset.from_pandas(
        valid_df,
        features=features, info=description)
    del valid_df
    gc.collect()
    test_df = pd.read_parquet(os.path.join(dataset_path, local_name, "text-to-sparql-test.parquet"), engine="pyarrow")
    test_df.reset_index(drop=True, inplace=True)
    test_df = test_df.astype({"id": "int16", "sparql_raw": "string", "sparql_annotated": "string",
                              "sparql_query": "string", "query_results": "string",
                              "complexity": "string", "complexity_description": "string"})
    dataset["test"] = Dataset.from_pandas(
        test_df,
        features=features, info=description)
    del test_df
    gc.collect()
    dataset.push_to_hub(repo_id, "with_limit", token=HF_TOKEN, max_shard_size="500MB")
    dataset.save_to_disk(os.path.join(dataset_path, local_name, "Instruct-to-SPARQL"), max_shard_size="500MB")


create_hf_dataset(base_dir, local_name="final_fq18_with_limit_10")
