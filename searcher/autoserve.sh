#!/bin/bash
set -e

if [ ! -d "searcher" ]; then
    echo 'Please run this script in the root directory of the project'
    exit 1
fi

# Usage: ./autoserve.sh <model_name> <version> <model_path>

model_name="$1"
version="$2"
model_path="$3"

torch-model-archiver -f \
    --model-name $model_name \
    --version 1.0 \
    --serialized-file $model_path \
    --handler searcher/deep_hashing_handler.py \
    --extra-files \
"$version/config.json,\
$version/remap_index_to_img_path_dict.json,\
$version/index.bin"
# module.zip"

mkdir -p model_store
mv $model_name.mar model_store/