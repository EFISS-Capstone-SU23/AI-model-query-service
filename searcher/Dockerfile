FROM python:3.11-bullseye as builder
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

RUN mkdir -p /app
WORKDIR /app

ARG VERSION=1.2.0
ARG YOLO_MODEL_PATH=torchscripts_models/yolo/yolo.pt

COPY ${YOLO_MODEL_PATH} /app/model/yolo.pt
COPY index/${VERSION} /app/model/
COPY searcher/config.properties searcher/deep_hashing_handler.py /app/
  
# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=model-image-retrieval \
  --version=1.0 \
  --handler=/app/deep_hashing_handler.py \
  --serialized-file=/app/model/yolo.pt \
  --extra-files "/app/model/config.json,/app/model/preprocessor_config.json,/app/model/pytorch_model.bin"

RUN mkdir -p /app/model_store && \
  mv model-image-retrieval.mar /app/model_store

COPY searcher/run.sh searcher/example.json /app/

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/run.sh"]