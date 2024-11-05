# Instruct-to-SPARQL: GitHub Repository

Welcome to the GitHub repository for the [Instruct-to-SPARQL](https://huggingface.co/datasets/PaDaS-Lab/Instruct-to-SPARQL) dataset! This repository contains the source code, data processing scripts, and examples for creating, fine-tuning, and evaluating models with the Instruct-to-SPARQL dataset.

## Table of Contents

- [Instruct-to-SPARQL Overview](#instruct-to-sparql-overview)
- [Dataset Details](#dataset-details)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Collection and Processing](#data-collection-and-processing)
- [Fine-tuning and Evaluation](#fine-tuning-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Instruct-to-SPARQL Overview

Instruct-to-SPARQL is a dataset that consists of pairs of Natural language instructions and SPARQL queries. The dataset is created by crawling Wikipedia pages and tutorials for real examples of WikiData SPARQL queries. The dataset has a total of 2.8k examples, split into train, validation, and test sets.

## Dataset Details

The dataset has the following features:

- **id**: A unique identifier for each example.
- **instructions**: A list of natural language instructions and questions.
- **sparql_raw**: The SPARQL query that was crawled and cleaned.
- **sparql_annotated**: The SPARQL query with annotations and Prefixes.
- **sparql_query**: The final SPARQL with Prefixes query used to retrieve data.
- **complexity**: A measure of the query's complexity: simple, medium, or complex.
- **complexity_description**: A description of the query's complexity.
- **query_results**: The results obtained from executing the SPARQL query.

## Repository Structure

The repository is organized as follows:

```
instruct-to-sparql/
├── data/
│   ├── raw/
│   ├── processed/
│   └── nl_generation/
├── scripts/
├── src/
├── sparql-wikidata.yml
└── README.md
```

- **data/**: Contains scripts and notebooks for data collection, cleaning, augmentation, and natural language generation.
- **src/**: Contains scripts for fine-tuning models, calculating metrics, and evaluating model performance.
- **scripts/**: Contains bash scripts for the experiments and model training.
- **env.yml**: Conda environment file with all required dependencies.

## Installation

To use this repository, you need to clone it and set up the required environment.

### Clone the Repository

```bash
git clone https://github.com/padas-lab-de/instruct-to-sparql.git
cd instruct-to-sparql
```

### Set Up the Environment

We recommend using a Conda environment. You can create the environment with the required dependencies using the `env.yml` file:

```bash
conda env create -f env.yml
conda activate instruct-to-sparql
```

## Data Collection and Processing

The dataset was created through several steps:

1. **Data Collection**: Crawling Wikipedia pages and tutorials for real examples of WikiData SPARQL queries.
2. **Data Cleaning**: Cleaning the collected data to ensure consistency and correctness.
3. **Data Augmentation**: Augmenting the dataset with additional examples to enhance diversity.
4. **Natural Language Generation**: Generating natural language instructions corresponding to the SPARQL queries.

### Data Collection a Cleaning

Scripts and notebooks for data collection can be found in the `data/` directory.


### Natural Language Generation & Augmentation

Scripts and notebooks for natural language generation can be found in the `data/nl_generation` directory.

## Fine-tuning and Evaluation

The repository includes code for fine-tuning models on the Instruct-to-SPARQL dataset and evaluating their performance.

### Fine-tuning

The `scripts/finetune.sh` script supports fine-tuning various models. Here are some example commands:

```bash
# Basic usage with default parameters
./scripts/finetune.sh --model_name="mistralai/Mistral-7B-Instruct-v0.3"

# Fine-tune Llama3 with annotated SPARQL
./scripts/finetune.sh \
    --model_name="meta-llama/Meta-Llama-3-8B-Instruct" \
    --annotated \
    --batch_size=2 \
    --accelerate="deepspeed-fp16" \
    --left_padding_side

# Fine-tune Mistral with custom parameters
./scripts/finetune.sh \
    --model_name="mistralai/Mistral-7B-Instruct-v0.3" \
    --batch_size=2 \
    --accelerate="deepspeed-bf16" \
    --left_padding_side
```

Available parameters:
- `--model_name`: The name/path of the model to fine-tune (default: "meta-llama/Meta-Llama-3-8B-Instruct")
- `--annotated`: Use annotated SPARQL queries for training
- `--batch_size`: Training batch size (default: 2)
- `--accelerate`: Acceleration strategy (options: "deepspeed-fp16", "deepspeed-bf16")
- `--use_peft`: Enable PEFT/LoRA training
- `--left_padding_side`: Use left padding instead of right padding

### Evaluation

The script `scripts/evaluate.sh` is used for evaluating model performance. To evaluate a model checkpoint, run:

```bash
./scripts/evaluate.sh --model_name MODEL_NAME --batch_size BATCH_SIZE --annotated --shots NUM_SHOTS
```


## Citation

If you use this dataset or code in your research, please cite it as follows:

```
@dataset{instruct_to_sparql,
  author = {Mehdi Ben Amor, Alexis Strappazon, Michael Granitzer, Előd Egyed-Zsigmond, Jelena Mitrovic},
  title = {Instruct-to-SPARQL},
  year = {2024},
  howpublished = {https://huggingface.co/datasets/PaDaS-Lab/Instruct-to-SPARQL},
  note = {A dataset of natural language instructions and corresponding SPARQL queries}
}
```

## License

This repository is licensed under the [Apache-2.0 license](LICENSE) license. See the `LICENSE` file for more details.

## Contact

For questions or comments about the dataset or repository, please contact [here](mailto:mehdi.benamor@uni-passau.de)
