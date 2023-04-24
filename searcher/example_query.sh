#!/bin/bash
cat <<EOF > option.json
{
    "topk": 10,
    "debug": true,
}
EOF

curl -X POST http://localhost:5000/predictions/image-retrieval-v1.0 -T option.json