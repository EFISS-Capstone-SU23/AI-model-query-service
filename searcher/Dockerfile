ARG VERSION=1.2.0
ARG YOLO_MODEL_PATH=torchscripts_models/yolo/yolo.pt

FROM python:3.11-bullseye as builder

# Avoid to write .pyc files on the import of source modules
ENV PYTHONDONTWRITEBYTECODE 1
# Enable fault handler
ENV PYTHONFAULTHANDLER 1

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
COPY searcher/requirements.txt requirements.txt
# RUN pip install -r requirements.txt
RUN pip3 install --disable-pip-version-check --no-cache-dir --target /install -r requirements.txt

RUN rm -rf /usr/local/lib/python3.11/site-packages/ && \
    ln -s /install /usr/local/lib/python3.11/site-packages

WORKDIR /app

COPY ${YOLO_MODEL_PATH} /app/model/yolo.pt
COPY index/${VERSION} /app/model/
COPY searcher/config.properties searcher/deep_hashing_handler.py /app/
  
# create model archive file packaging model artifacts and dependencies
RUN /usr/local/lib/python3.11/site-packages/bin/torch-model-archiver -f \
  --model-name=model-image-retrieval \
  --version=1.0 \
  --handler=/app/deep_hashing_handler.py \
  --serialized-file=/app/model/yolo.pt \
  --extra-files "/app/model/config.json,/app/model/preprocessor_config.json,/app/model/pytorch_model.bin"

RUN mkdir -p /app/model_store && \
  mv model-image-retrieval.mar /app/model_store

COPY searcher/run.sh searcher/example.json /app/

FROM python:3.11-bullseye

# Force stdin, stdout and stderr to be totally unbuffered
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages

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

WORKDIR /app

COPY --from=builder /app/run.sh /app/config.properties /app/example.json /app/
COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/model_store /app/model_store

ENTRYPOINT [ "/bin/bash" ]
CMD [ "/app/run.sh"]