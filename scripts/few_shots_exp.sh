#!/usr/bin/bash
# -*- coding: utf-8 -*-
# set -euo pipefail

CONDA_DIR=/opt/conda
CONDA_ENV_NAME=sparql-wikidata-vllm

export PATH="$CONDA_DIR/condabin:$PATH"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

NUM_SHOTS=(0 3 5)

# Hyperparameters
# Default values for named arguments
ANNOTATED_GEN=false
BATCH_SIZE=32
DEVICES=0
AGENT_ID=0
DATASET_PATH="PaDaS-Lab/Instruct-to-SPARQL"
SUBSET="default"
USE_FAST=false
NUM_SAMPLES=-1
PROJECT_NAME="sft-sparql"
PROMPT_STYLE="chatml"
SEED=42
MAX_NEW_TOKENS=1024
TOP_K=20
TEMPERATURE=0.7
TOP_P=0.7
NUM_RETURN_SEQUENCES=1
DO_SAMPLE=true

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --annotated_gen)
    ANNOTATED_GEN=true
    ;;
  --agent_id=*)
    AGENT_ID="${1#*=}"
    ;;
  --batch_size=*)
    BATCH_SIZE="${1#*=}"
    ;;
  --devices=*)
    DEVICES="${1#*=}"
    ;;
  --dataset_path=*)
    DATASET_PATH="${1#*=}"
    ;;
  --subset=*)
    SUBSET="${1#*=}"
    ;;
  --use_fast)
    USE_FAST=true
    ;;
  --num_samples=*)
    NUM_SAMPLES="${1#*=}"
    ;;
  --project_name=*)
    PROJECT_NAME="${1#*=}"
    ;;
  --prompt_style=*)
    PROMPT_STYLE="${1#*=}"
    ;;
  --seed=*)
    SEED="${1#*=}"
    ;;
  --max_new_tokens=*)
    MAX_NEW_TOKENS="${1#*=}"
    ;;
  --top_k=*)
    TOP_K="${1#*=}"
    ;;
  --temperature=*)
    TEMPERATURE="${1#*=}"
    ;;
  --top_p=*)
    TOP_P="${1#*=}"
    ;;
  --num_return_sequences=*)
    NUM_RETURN_SEQUENCES="${1#*=}"
    ;;
  --api_url=*)
    API_URL="${1#*=}"
    ;;
  --api_key=*)
    API_KEY="${1#*=}"
    ;;
  --do_sample)
    DO_SAMPLE=true
    ;;
  --models=*)
    IFS=',' read -ra FEW_SHOT_MODELS <<< "${1#*=}"
    ;;
  --num_shots=*)
    IFS=',' read -ra NUM_SHOTS <<< "${1#*=}"
    ;;
  *)
    printf "Invalid argument: %s\n" "$1"
    exit 1
    ;;
  esac
  shift
done


# Hyperparameters
DATASET_PATH="PaDaS-Lab/Instruct-to-SPARQL"
SUBSET=("default")    # "default", "with_limit"
PROMPT_STYLE="chatml" # "prompts"
SEED=42

if [ "$ANNOTATED_GEN" = true ]; then
  PROJECT_NAME="sft-sparql-annotated --annotated_gen"
else
  PROJECT_NAME="sft-sparql"
fi

REPO=/data/instruct-to-sparql
export PYTHONPATH="${PYTHONPATH}:$REPO"
export CUDA_VISIBLE_DEVICES=$DEVICES
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export INFERENCE_API_KEY=$API_KEY
export INFERENCE_API_URL=$API_URL

# List of valid model names
VALID_MODELS=(
  "gpt-4o"
  "gpt-4o-mini"
  "gpt-3.5-turbo-1106"
  "llama3.1"
  "qwen2.5"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "Qwen/Qwen2.5-0.5B-Instruct"
  "PaDaS-Lab/Llama3-8B-SPARQL"
  "PaDaS-Lab/Mistral-7B-v0.3-SPARQL"
  "PaDaS-Lab/Llama3-8B-SPARQL-annotated"
  "PaDaS-Lab/Mistral-7B-v0.3-SPARQL-annotated"
  "/data/llms/sft-sparql/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-False"
  "/data/llms/sft-sparql-annotated/Qwen2.5-0.5B-Instruct-PaDaS-Lab-default-sft-cosine-chatml-annotated-True"
)

# Function to check if a model is valid
is_valid_model() {
  local model="$1"
  for valid_model in "${VALID_MODELS[@]}"; do
    if [[ "$model" == "$valid_model" ]]; then
      return 0
    fi
  done
  return 1
}

# Validate input models
for model in "${FEW_SHOT_MODELS[@]}"; do
  if ! is_valid_model "$model"; then
    echo "Error: Invalid model name '$model'"
    exit 1
  fi
done

# If no models were provided, use all valid models
if [ ${#FEW_SHOT_MODELS[@]} -eq 0 ]; then
  FEW_SHOT_MODELS=("${VALID_MODELS[@]}")
fi

# Iterate over all few-shot models
for MODEL_NAME in "${FEW_SHOT_MODELS[@]}"; do
  echo "Evaluating model: ${MODEL_NAME}"
  
  # Run evaluation for each number of shots
  for N_SHOTS in "${NUM_SHOTS[@]}"; do
    echo "Running evaluation for ${MODEL_NAME} with ${N_SHOTS} shots"
    python src/eval.py \
      --model_name "${MODEL_NAME}" \
      --dataset_path "${DATASET_PATH}" \
      --subset "${SUBSET}" \
      --use_fast "${USE_FAST}" \
      --annotated_gen "${ANNOTATED_GEN}" \
      --num_samples "${NUM_SAMPLES}" \
      --project_name "${PROJECT_NAME}" \
      --prompt_style "${PROMPT_STYLE}" \
      --n_shots "${N_SHOTS}" \
      --agent_id "${AGENT_ID}" \
      --batch_size "${BATCH_SIZE}" \
      --seed "${SEED}" \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --top_k "${TOP_K}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --num_return_sequences "${NUM_RETURN_SEQUENCES}" \
      ${DO_SAMPLE:+--do_sample}
  done
done