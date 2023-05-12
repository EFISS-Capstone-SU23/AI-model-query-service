FROM python:3.9-bullseye
RUN mkdir -p /install
WORKDIR /install

# install java
RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    openjdk-11-jre-headless \
    git \
    wget \
    g++ \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# install dependencies
COPY searcher/requirements.txt /install
RUN pip install -r requirements.txt

RUN pip install pyyaml captum

ARG VERSION=1.2.0
ARG MODEL_NAME=relahash-medium-64bits

RUN mkdir -p /app
WORKDIR /app

COPY searcher/config.properties searcher/deep_hashing_handler.py /app/
COPY index/${VERSION} /app/index/${VERSION}
COPY torchscripts_models/${MODEL_NAME}.pt /app/torchscripts_models/${MODEL_NAME}.pt
  
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

COPY searcher/run.sh /app/

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/run.sh"]