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

    1.1. Get `database_info.txt` file

    ```bash
    bash indexer/extract_datalake.sh /path/to/datalake
    ```

    1.2. Run [indexer/main.py](indexer/main.py)

    ```bash
    # database_info.txt
    data/abc.com/69_abc_com.jpg
    data/abc.com/42_abc_com.jpg

    python indexer/main.py \
        --database database_info.txt \
        --model_path torchscripts_models/relahash-medium-64bits.pt \
        --device cuda:0 \
        --new_index_database_version 1.5.0
    ```


2. API: `/api/reindex` _(still underdevelopment)_

    Note: this should be change to cronjob instead of calling Admin API

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
