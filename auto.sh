#!/bin/bash

# log current time
echo ----- $(date) -------

cd /home/saplab/thaiminhpv/EFISS/AI-model-query-service

echo "Start syncing data from GCS"
gsutil -m rsync -r gs://efiss/data/product_images /home/saplab/thaiminhpv/relahash/rsi-prototype/data/shopee_29-7/product_images > /dev/null 2>&1
echo Done, begin indexing

bash indexer/increase_version.sh
bash indexer/reindex.sh

echo Done!