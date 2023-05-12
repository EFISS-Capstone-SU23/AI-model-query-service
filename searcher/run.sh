#!/bin/bash
. /root/.bashrc
exec torchserve --start --model-store /app/model_store --models image-retrieval-v1.0=${MODEL_NAME}.mar --ts-config /app/config.properties --foreground