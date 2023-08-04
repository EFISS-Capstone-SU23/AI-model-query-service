FROM python:3.11-bullseye
RUN mkdir -p /install
WORKDIR /install

# install java
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    build-essential \
    cmake \
    ffmpeg libsm6 libxext6 libgl1 \
    git \
    wget \
    g++ \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY searcher/requirements.txt /install
RUN pip install -r requirements.txt

ARG VERSION=1.2.0
ARG MODEL_NAME=relahash-medium-64bits
ARG YOLO_MODEL_PATH=torchscripts_models/yolo/yolo.pt

RUN mkdir -p /app
WORKDIR /app

COPY index/${VERSION} /app/index/${VERSION}
# COPY torchscripts_models/${MODEL_NAME}.pt /app/torchscripts_models/model-image-retrieval.pt
COPY ${YOLO_MODEL_PATH}.pt /app/torchscripts_models/yolo.pt
COPY searcher/config.properties searcher/deep_hashing_handler.py /app/
  
# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=model-image-retrieval \
  --version=1.0 \
  --handler=/app/deep_hashing_handler.py \
  --extra-files "/app/index/${VERSION}/config.json,/app/index/${VERSION}/remap_index_to_img_path_dict.json,/app/index/${VERSION}/index.bin,/app/torchscripts_models/yolo.pt"
#   --serialized-file=/app/torchscripts_models/model-image-retrieval.pt \

RUN mkdir -p /app/model_store && \
  mv model-image-retrieval.mar /app/model_store

COPY searcher/run.sh searcher/example.json /app/

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/run.sh"]