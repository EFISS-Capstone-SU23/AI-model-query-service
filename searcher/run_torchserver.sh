#!/bin/bash
set -x
set -e

# if [ ! -d "searcher" ]; then
#     echo 'Please run this script in the root directory of the project'
#     exit 1
# fi

# find . | egrep "\.(py)$" | zip -@ module.zip
# mv module.zip searcher/
# cd searcher

model_name="relahash-medium-64bits"
version="index/1.2.0"

torch-model-archiver -f \
    --model-name $model_name \
    --version 1.0 \
    --serialized-file "/media/thaiminhpv/Storage/MinhFileServer/Public-Filebrowser/RelaHash_weights/torchscripts/relahash_tf_efficientnetv2_b3_relahash_64_deepfashion2_200_0.0005_adam.pt" \
    --handler deep_hashing_handler.py \
    --extra-files \
"$version/config.json,\
$version/remap_index_to_img_path_dict.json"
# module.zip"

mkdir -p model_store
mv $model_name.mar model_store/
# rm -v module.zip

torchserve --stop

torchserve --start \
--model-store model_store \
--ts-config config.properties \
--ncs --models image-retrieval-v1.0=$model_name.mar

