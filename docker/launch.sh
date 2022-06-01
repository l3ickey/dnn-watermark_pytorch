#!/bin/bash

TAG='funabiki/dnn-watermark_pytorch_pytorch:1.11.0-cuda11.3-cudnn8-devel'
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"
DATASET_DIR="${PROJECT_DIR}/../dataset/cifar-10-batches-py"

# build
cd "$(dirname "${0}")/.." || exit
DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} docker

# run
docker run -it --rm \
  --shm-size=8g \
  --gpus all \
  -v "${PROJECT_DIR}:/workspace" \
  -v "${DATASET_DIR}:/root/dataset/cifar-10-batches-py" \
  -w "/workspace" \
  "${TAG}"