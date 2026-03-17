#!/bin/bash

TPU_NAME="$1"
shift
SSH_FLAGS='-A -o ForwardAgent=yes'
COMMANDS="if [ ! -d \"power\" ]; then git clone git@github.com:sdbuch/power; fi \
    && export HYDRA_FULL_ERROR=1 \
    && export LIBTPU_INIT_ARGS='--xla_memory_scheduler=default' \
    && export WANDB_ENTITY='$WANDB_ENTITY' \
    && export WANDB_API_KEY='$WANDB_API_KEY' \
    && export HF_TOKEN='$HF_TOKEN' \
    && cd power \
    && git fetch \
    && git checkout -f main \
    && git pull \
    && uv sync --extra tpu \
    && uv run python $@"

gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
  --ssh-flag="$SSH_FLAGS" \
  --command="$COMMANDS" \
  --worker=all
