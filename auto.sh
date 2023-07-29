#!/bin/bash

# log current time
echo ----- $(date) -------

cd /home/saplab/thaiminhpv/EFISS/AI-model-query-service
bash indexer/increase_version.sh
bash indexer/reindex.sh

echo "Start syncing data to GCS"
gsutil -m rsync -r /home/saplab/thaiminhpv/relahash/rsi-prototype/data/shopee_29-7/product_images gs://efiss/data/product_images > /dev/null 2>&1
echo Done