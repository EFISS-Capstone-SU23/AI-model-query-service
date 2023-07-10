#!/bin/bash
set -x
set -e

if [ ! -d "searcher" ]; then
    echo 'Please run this script in the root directory of the project'
    exit 1
fi

# find . | egrep "\.(py)$" | zip -@ module.zip
# mv module.zip searcher/
# cd searcher

model_name="orthocos_medium_few_shot_10ep_aug_4096bit_shopee_category_phrase2"
version="index/2.0.1"

torch-model-archiver -f \
    --model-name $model_name \
    --version 1.0 \
    --serialized-file "torchscripts_models/orthocos_medium_few_shot_10ep_aug_4096bit_shopee_category_phrase2.pt" \
    --handler searcher/deep_hashing_handler.py \
    --extra-files \
"$version/config.json,\
$version/remap_index_to_img_path_dict.json,\
$version/index.bin"
# module.zip"

mkdir -p model_store
mv $model_name.mar model_store/
# rm -v module.zip

torchserve --stop

CUDA_VISIBLE_DEVICES=0 torchserve --start \
--model-store model_store \
--ts-config searcher/config.properties \
--ncs --models image-retrieval-v1.0=$model_name.mar

