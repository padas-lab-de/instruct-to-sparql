from datasets import load_dataset
from argparse import ArgumentParser
import json
from sentence_transformers import SentenceTransformer
import torch

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="PaDaS-Lab/Instruct-to-SPARQL")
    parser.add_argument("--subset", type=str, default="default")
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()

def encode_and_search(model, train_data, test_data, k):
    # Select the first instruction from each list
    train_instructions = [instrs[0] for instrs in train_data]
    test_instructions = [instrs[0] for instrs in test_data]
    
    train_embeddings = model.encode(train_instructions)
    test_embeddings = model.encode(test_instructions)
    
    similarities = model.similarity(test_embeddings, train_embeddings)
    
    top_k_indices = torch.argsort(similarities, dim=1, descending=True)[:, :k]
    
    return top_k_indices.tolist()

def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_path, name=args.subset)
    model = SentenceTransformer('BAAI/bge-small-en-v1.5')

    # Encode and search using instructions
    train_instructions = dataset["train"].filter(lambda x: len(x["sparql_query"].split(" ")) < 300)["instructions"]
    test_instructions = dataset["test"]["instructions"]
    top_k = encode_and_search(model, train_instructions, test_instructions, args.k)

    # Prepare results
    results = {}
    for i, indices in enumerate(top_k):
        results[i] = indices

    # Save to json
    with open("few_shots.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    main()
