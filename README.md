# Instruct-to-SPARQL: GitHub Repository

Welcome to the GitHub repository for the Instruct-to-SPARQL dataset! This repository contains the source code, data processing scripts, and examples for creating, fine-tuning, and evaluating models with the Instruct-to-SPARQL dataset.

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
- **instructions**: A sequence of natural language instructions.
- **sparql_raw**: The raw SPARQL query as found in the source.
- **sparql_annotated**: The SPARQL query with annotations.
- **sparql_query**: The final SPARQL query used to retrieve data.
- **complexity**: A measure of the query's complexity.
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
git clone https://github.com/your_username/instruct-to-sparql.git
cd instruct-to-sparql
```

### Set Up the Environment

We recommend using a Conda environment. You can create the environment with the required dependencies using the `env.yml` file:

```bash
conda env create -f sparql-wikidata.yml
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

The `scripts/` directory contains scripts for fine-tuning and evaluating models. For examplae, to fine-tune the following models, run:

* For the model [Llama3-8B-SPARQL-annotated](https://huggingface.co/PaDaS-Lab/Llama3-8B-SPARQL-annotated)
    ```bash
    ./scripts/llama3_sparql.sh --annotated --batch_size=2 --accelerate="deepspeed-fp16" --left_padding_side
    
    ```
* For the model [Mistral-7B-v0.3-SPARQL](https://huggingface.co/PaDaS-Lab/Mistral-7B-v0.3-SPARQL)
    ```bash
    ./scripts/mistral_sparql.sh --batch_size=2 --accelerate="deepspeed-bf16" --left_padding_side
    ```
### Evaluation

The script `scripts/evaluate.sh` is used for evaluating model performance. To evaluate a model checkpoint, run:

```bash
./scripts/evaluate.sh --model_name MODEL_NAME --batch_size BATCH_SIZE --annotated
```

## Metrics

The performance of models on the Instruct-to-SPARQL dataset is evaluated using a combination of machine translation metrics and execution result metrics.

### Machine Translation Metrics

1. **[BLEU](https://aclanthology.org/P02-1040.pdf) (Bilingual Evaluation Understudy)**

   BLEU measures the similarity between the generated SPARQL query and a reference SPARQL query by calculating n-gram precision. It ranges from 0 to 1, where 1 indicates a perfect match.

   Formula:
   ```math
   \text{BLEU} = \exp \left( \min\left(0, 1 - \frac{\text{len(ref)}}{\text{len(gen)}}\right) + \sum_{n=1}^{N} w_n \log p_n \right)
   ```
   where $p_n$ is the precision of n-grams, $w_n$ are weights, and $\text{len(ref)}$ and $\text{len(gen)}$ are the lengths of the reference and generated queries, respectively.

2. **[ROUGE](https://aclanthology.org/W04-1013.pdf) (Recall-Oriented Understudy for Gisting Evaluation)**

   ROUGE measures the overlap between the generated SPARQL query and the reference SPARQL query, focusing on recall. The most commonly used variants are ROUGE-N (n-gram recall) and ROUGE-L (longest common subsequence).

   ROUGE-N Formula:
   ```math
   \text{ROUGE-N} = \frac{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}_{match}(gram_n)}{\sum_{S \in \text{References}} \sum_{gram_n \in S} \text{Count}(gram_n)}
   ```

   ROUGE-L Formula:
   ```math
   \text{ROUGE-L} = \frac{LCS(X,Y)}{\text{len}(Y)}
   ```
   where $LCS(X, Y)$ is the length of the longest common subsequence between the reference $X$ and the generated query $Y$.

### Execution Results Metrics

1. **[Overlap Coefficient](https://en.wikipedia.org/wiki/Overlap_coefficient)**

   The Overlap Coefficient measures the similarity between the sets of results returned by the target and generated SPARQL queries. It is defined as the size of the intersection divided by the size of the smaller set.

   Formula:
   ```math
   \text{Overlap Coefficient} = \frac{|A \cap B|}{\min(|A|, |B|)}
   ```
   where $A$ and $B$ are the sets of results from the target and generated queries, respectively.

2. **[Jaccard Similarity](https://en.wikipedia.org/wiki/Jaccard_index)**

   The Jaccard Similarity measures the similarity between the sets of results returned by the target and generated SPARQL queries. It is defined as the size of the intersection divided by the size of the union.

   Formula:
   ```math
   \text{Jaccard Similarity} = \frac{|A \cap B|}{|A \cup B|}
   ```
   where $A$ and $B$ are the sets of results from the target and generated queries, respectively.

## Citation

If you use this dataset or code in your research, please cite it as follows:

```
@dataset{instruct_to_sparql,
  author = {Mehdi Ben Amor, Alexis Strappazon, Michael Granitzer, Jelena Mitrovic},
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
