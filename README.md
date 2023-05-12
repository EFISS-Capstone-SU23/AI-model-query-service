# TorchServe API

API: predictions/image-retrieval-v1.0

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

# Admin API

Reindex the database

1. By running [indexer/main.py](indexer/main.py) directly _(recommended)_

```bash
# database_info.txt
data/abc.com/69_abc_com.jpg
data/abc.com/42_abc_com.jpg

python indexer/main.py \
    --database database_info.txt \
    --model_path torchscripts_models/relahash_tf_efficientnetv2_b3_relahash_64_deepfashion2_200_0.0005_adam.pt \
    --device cuda:0 \
    --new_index_database_version 1.4.0
```


2. API: /api/reindex _(still underdevelopment)_

Input: multipart/form-data

```json
{
    "new_index_database_version": "1.2.0", // version of the new index database
    "mode": "default", // ['default', 'unnecessary fast']
    "model_path": "model_name/001/"
}
```

Output:

```json
{
    "result": "success",
    "previous_index_database_version": "1.1.0", // version of the previous index database
    "index_database_version": "1.2.0", // version of the new index database
    "timestamp": "2020-05-02 12:00:00",
}
```

# Build

1. Build CPU docker image

```bash
docker build -t efiss-ai:latest -t efiss-ai:1.0.0-cpu -f searcher/Dockerfile .
```

2. Build GPU docker image

```bash
docker build -t efiss-ai:latest-cuda -t efiss-ai:1.0.0-cuda -f searcher/Dockerfile.cuda .
```
