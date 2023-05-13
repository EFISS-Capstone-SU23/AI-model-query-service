# AI model service for EFISS

## TorchServe API

API: `https://ai.efiss.tech/predictions/image-retrieval-v1.0`

Request:

```json
{
    "top_k": 10,
    "image": "<base64 encoded image>"
}
```

Response

```json
{
    "index_database_version": "1.2.0", // version of the index database
    "relevant": [
        "data/abc.com/69_abc_com.jpg",
        "data/abc.com/42_abc_com.jpg"
    ],  // file path of relevant images, sorted by relevance from most relevant to least relevant
}
```

### Debug mode

Request:

```json
{
    "top_k": 10,
    "image": "<base64 encoded image>",
    "debug": true,
}
```

Response

```json
{
    "index_database_version": "1.2.0", // version of the index database
    "relevant": [
        "data/abc.com/69_abc_com.jpg",
        "data/abc.com/42_abc_com.jpg"
    ],  // file path of relevant images, sorted by relevance from most relevant to least relevant
    "distances": [
        2,
        50
    ]
}
```

## Reindex the database

1. Update version by editing .env

```bash
cp .env.example .env
```

2. Just run

```bash
bash indexer/reindex.sh
```

It will gather images data, index them, dockerize them, push to GCR, and re-run the GPU service locally.

## Build

1. Build CPU docker image

```bash
docker build -t efiss-ai:latest \
    -t efiss-ai:1.0.0-cpu \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:1.0.0-cpu \
    -f searcher/Dockerfile .

docker push asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:1.0.0-cpu
```

2. Build GPU docker image

```bash
docker build -t efiss-ai:latest-cuda \
    -t efiss-ai:1.0.0-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:latest-cuda \
    -t asia-southeast1-docker.pkg.dev/even-acumen-386115/efiss/efiss-ai:1.0.0-cuda \
    -f searcher/Dockerfile.cuda .
```
