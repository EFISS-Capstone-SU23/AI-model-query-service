#!/bin/bash
set -e

if [ ! -d "searcher" ]; then
    echo 'Please run this script in the root directory of the project'
    exit 1
fi

# import env variables
if [ ! -f .env ]; then
    echo "Please create a .env file by using `cp .env.example .env`"
    exit 1
fi
set -o allexport
source .env
set +o allexport

# VERSION="1.5.0"
if [ -z "$VERSION" ]; then
    echo "Please provide a version number"
    exit 1
fi

bash indexer/extract_datalake.sh /media/saplab/Data_Win/RSI_Do_An/AnhND/Dynamic-Crawler-Tool/output

set -x

python indexer/main.py \
    --database database_info.txt \
    --model_path torchscripts_models/relahash-medium-64bits.pt \
    --device cuda:0 \
    --batch_size 384 \
    --num_workers 16 \
    --new_index_database_version $VERSION

docker build -t efiss-ai:latest \
    -t efiss-ai:$VERSION-cpu \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$VERSION-cpu \
    --build-arg MODEL_NAME=relahash-medium-64bits \
    --build-arg VERSION=$VERSION \
    -f searcher/Dockerfile .

docker push asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$VERSION-cpu
docker push asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest


docker build -t efiss-ai:latest-cuda \
    -t efiss-ai:$VERSION-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:$VERSION-cuda \
    --build-arg MODEL_NAME=relahash-medium-64bits \
    --build-arg VERSION=$VERSION \
    -f searcher/Dockerfile.cuda .

docker compose up -d