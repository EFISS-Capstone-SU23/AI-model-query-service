#!/bin/bash

# log current time
echo ----- $(date) -------

cd /home/saplab/thaiminhpv/EFISS/AI-model-query-service
bash indexer/increase_version.sh
bash indexer/reindex.sh

echo "Start syncing data to GCS"
gsutil -m rsync -r /media/saplab/Data_Win/RSI_Do_An/AnhND/Dynamic-Crawler-Tool/output gs://efiss/data/output > /dev/null 2>&1
echo Done