#!/bin/bash
. /root/.bashrc
(
    sleep 10
    echo "Warming up..."
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T /app/example.json
    echo "Done warming up."
) &
exec torchserve --start --model-store /app/model_store --models image-retrieval-v1.0=${MODEL_NAME}.mar --ts-config /app/config.properties --foreground