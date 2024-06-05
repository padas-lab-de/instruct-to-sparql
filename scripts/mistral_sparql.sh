#!/usr/bin/bash
# -*- coding: utf-8 -*-
#set -exuo pipefail

CONDA_DIR=/opt/conda
CONDA_ENV_NAME=sparql-wikidata

# This script assumes the following:
# (1) a conda environment named $CONDA_ENV_NAME
# If using venv, you can remove the conda stuff and just activate the venv directly
#set +x
export PATH="$CONDA_DIR/condabin:$PATH"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Hyperparameters
# Default values for named arguments
ANNOTATED=false
BATCH_SIZE=2
ACCELERATE="deepspeed-bf16"
USE_PEFT=false
PADDING_SIDE=false

# Parse named arguments
while [ $# -gt 0 ]; do
  case "$1" in
  --annotated)
    ANNOTATED=true
    ;;
  --batch_size=*)
    BATCH_SIZE="${1#*=}"
    ;;
  --accelerate=*)
    ACCELERATE="${1#*=}"
    ;;
  --use_peft)
    USE_PEFT=true
    ;;
  --left_padding_side)
    PADDING_SIDE=true
    ;;
  *)
    printf "Invalid argument: %s\n" "$1"
    exit 1
    ;;
  esac
  shift
done

# Hyperparameters
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH="PaDaS-Lab/Instruct-to-SPARQL"
SUBSET=("default" "with_limit") # "default", "with_limit"
PROMPT_STYLE="chatml"           # "prompts"
SEED=42
TOTAL_STEPS=-1
EVAL_INTERVAL=1

if [ "$ANNOTATED" = true ]; then
  PROJECT_NAME="sft-sparql-annotated --annotated_gen"
else
  PROJECT_NAME="sft-sparql"
fi

EPOCHS=3
BATCH_SIZE=$BATCH_SIZE
ACCELERATE=$ACCELERATE
NUM_INSTRUCT=3

if [ "$PADDING_SIDE" = true ]; then
  FORCE_PADDING_SIDE="--force_padding_side"
else
  FORCE_PADDING_SIDE=""
fi

REPO=/data/llm-sparql-wikidata
export PYTHONPATH="${PYTHONPATH}:$REPO"
export CUDA_VISIBLE_DEVICES=0,1,3,4,5,6

if [ "$ACCELERATE" == "deepspeed-bf16" ]; then
  CONFIG_FILE=configs/accelerate/sparql-zero2-bf16.yaml
  AUTO_FIND_BATCH_SIZE="--bf16"
elif [ "$ACCELERATE" == "deepspeed-fp16" ]; then
  CONFIG_FILE=configs/accelerate/zero2-fp16.yaml
  AUTO_FIND_BATCH_SIZE="--fp16"
else
  CONFIG_FILE=configs/accelerate/ddp.yaml
  AUTO_FIND_BATCH_SIZE="--bf16 --auto_find_batch_size"
fi

for SUBSET_ in "${SUBSET[@]}"; do
  if [ "$USE_PEFT" = true ]; then
    echo "Training with LoRa on dataset subset: Text to Sparql $SUBSET_"
    accelerate launch --config_file $CONFIG_FILE src/trl_sft.py --model_name $MODEL_NAME --dataset_path $DATASET_PATH \
      --seed $SEED --total_steps $TOTAL_STEPS --eval_interval $EVAL_INTERVAL --project_name $PROJECT_NAME \
      --prompt_style $PROMPT_STYLE --num_train_epochs $EPOCHS --per_device_train_batch_size $BATCH_SIZE \
      --subset $SUBSET_ $AUTO_FIND_BATCH_SIZE --num_instruction $NUM_INSTRUCT --use_peft $FORCE_PADDING_SIDE
  else
    echo "Training without LoRa on dataset subset: Text to Sparql $SUBSET_"
    accelerate launch --config_file $CONFIG_FILE src/trl_sft.py --model_name $MODEL_NAME --dataset_path $DATASET_PATH \
      --seed $SEED --total_steps $TOTAL_STEPS --eval_interval $EVAL_INTERVAL --project_name $PROJECT_NAME \
      --prompt_style $PROMPT_STYLE --num_train_epochs $EPOCHS --per_device_train_batch_size $BATCH_SIZE \
      --subset $SUBSET_ $AUTO_FIND_BATCH_SIZE --num_instruction $NUM_INSTRUCT $FORCE_PADDING_SIDE
  fi
done
