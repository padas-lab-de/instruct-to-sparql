#!/usr/bin/bash
# -*- coding: utf-8 -*-
#set -exuo pipefail

CONDA_DIR=/opt/conda
CONDA_ENV_NAME=sparql-wikidata-vllm

# This script assumes the following:
# (1) a conda environment named $CONDA_ENV_NAME
# (2) It is being run from the $TRLX_DIR directory
# If using venv, you can remove the conda stuff and just activate the venv directly
#set +x
export PATH="$CONDA_DIR/condabin:$PATH"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Hyperparameters
# Default values for named arguments
ANNOTATED=false
BATCH_SIZE=64
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"
DEVICES=0,1,3,4,5,6
AGENT_ID=0
SHOTS=3

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --annotated)
    ANNOTATED=true
    ;;
  --agent_id=*)
    AGENT_ID="${1#*=}"
    ;;
  --api_url=*)
    API_URL="${1#*=}"
    ;;
  --api_key=*)
    API_KEY="${1#*=}"
    ;;
  --batch_size=*)
    BATCH_SIZE="${1#*=}"
    ;;
  --model_name=*)
    MODEL_NAME="${1#*=}"
    ;;
  --devices=*)
    DEVICES="${1#*=}"
    ;;
  --shots=*)
    SHOTS="${1#*=}"
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

if [ "$ANNOTATED" = true ]; then
  PROJECT_NAME="sft-sparql-annotated --annotated_gen"
else
  PROJECT_NAME="sft-sparql"
fi

REPO=/data/instruct-to-sparql
export PYTHONPATH="${PYTHONPATH}:$REPO"
export CUDA_VISIBLE_DEVICES=$DEVICES
export INFERENCE_API_KEY=$API_KEY
export INFERENCE_API_URL=$API_URL
export VLLM_WORKER_MULTIPROC_METHOD=spawn

for SUBSET_ in "${SUBSET[@]}"; do
  echo "Evaluating $MODEL_NAME on dataset subset: $SUBSET_"
  python src/eval.py --model_name $MODEL_NAME --dataset_path $DATASET_PATH \
    --seed $SEED --project_name $PROJECT_NAME \
    --prompt_style $PROMPT_STYLE --batch_size $BATCH_SIZE \
    --subset $SUBSET_ --agent_id $AGENT_ID --n_shots $SHOTS
done
