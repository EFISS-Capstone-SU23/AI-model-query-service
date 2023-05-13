#!/bin/bash
set -e

if [ ! -d "searcher" ]; then
    echo 'Please run this script in the root directory of the project'
    exit 1
fi

version="1.5.0"

bash indexer/extract_datalake.sh /media/saplab/Data_Win/RSI_Do_An/AnhND/Dynamic-Crawler-Tool/output

set -x

python indexer/main.py \
    --database database_info.txt \
    --model_path torchscripts_models/relahash-medium-64bits.pt \
    --device cuda:0 \
    --batch_size 384 \
    --num_workers 16 \
    --new_index_database_version $version

docker build -t efiss-ai:latest \
    -t efiss-ai:$version-cpu \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$version-cpu \
    --build-arg MODEL_NAME=relahash-medium-64bits \
    --build-arg VERSION=$version \
    -f searcher/Dockerfile .

docker push asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$version-cpu
docker push asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest


docker build -t efiss-ai:latest-cuda \
    -t efiss-ai:$version-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$version-cuda \
    --build-arg MODEL_NAME=relahash-medium-64bits \
    --build-arg VERSION=$version \
    -f searcher/Dockerfile.cuda .