FROM python:3.8-bullseye
# Install torchserve dependencies
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/pytorch/serve/ && cd serve && \
    python ./ts_scripts/install_dependencies.py

# install dependencies
RUN mkdir -p /install
WORKDIR /install
COPY searcher/requirements.txt /install
RUN pip install -r requirements.txt

ARG VERSION=1.2.0
ARG MODEL_NAME=relahash-medium-64bits

RUN mkdir -p /app
WORKDIR /app

COPY searcher/config.properties searcher/deep_hashing_handler.py /app/
COPY index/${VERSION} /app/index/${VERSION}
COPY torchscripts_models/ /app/torchscripts_models
  
# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=$MODEL_NAME \
  --version=1.0 \
  --serialized-file=/app/torchscripts_models/${MODEL_NAME}.pt \
  --handler=/app/deep_hashing_handler.py \
  --extra-files "/app/index/${VERSION}/config.json,/app/index/${VERSION}/remap_index_to_img_path_dict.json,/app/index/${VERSION}/index.bin"

RUN mkdir -p /app/model_store && \
  mv ${MODEL_NAME}.mar /app/model_store

ENV MODEL_NAME=relahash-medium-64bits

# ENTRYPOINT [ "/bin/bash", "-c" ]
CMD torchserve --start --model-store /app/model_store --models image-retrieval-v1.0=${MODEL_NAME}.mar --ts-config /app/config.properties