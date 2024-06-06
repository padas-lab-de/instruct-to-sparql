#!/bin/bash

# Fine-tuning script for a SPARQL model
MODEL_NAME="Llama3-8B-SPARQL-annotated"
BATCH_SIZE=2
ACCELERATE="deepspeed-fp16"

echo "Starting fine-tuning for $MODEL_NAME"

# Activate environment
source activate instruct-to-sparql

# Run fine-tuning
./scripts/llama3_sparql.sh --annotated --batch_size=$BATCH_SIZE --accelerate=$ACCELERATE --left_padding_side

echo "Fine-tuning completed for $MODEL_NAME"
