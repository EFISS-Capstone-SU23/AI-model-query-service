#!/bin/bash

# This script extracts the data from the datalake and dump it into a database_info.txt
# Usage: ./extract_datalake.sh <path_to_datalake_folder>
# Datalake folder should contain the following structure:
# datalake/
# ├── boo.vn
# │   ├── <id>_boo.vn.jpg
# │   ├── <id>_boo.vn.jpg
# ├── canifa.com
# │   ├── <id>_canifa.com.jpg
# │   ├── <id>_canifa.com.jpg
# The final database_info.txt file should contains only absolute path to the image each line.
# /home/foo/bar/datalake/boo.vn/00000000_boo.vn.jpg
# /home/foo/bar/datalake/boo.vn/00000001_boo.vn.jpg

if [ $# -ne 1 ]; then
    echo "Please provide the path to the datalake folder"
    exit 1
fi

# set -x
set -e

datadir=$1

if [ ! -d "$datadir" ]; then
    echo "Please provide the path to the datalake folder"
    exit 1
fi

rm -v database_info.txt || true

for domain in $(ls $datadir); do
    for img in $(ls $datadir/$domain); do
        absolute_path=$(realpath $datadir/$domain/$img)
        echo $absolute_path >> database_info.txt
    done
done

echo "Done"
echo "Total number of domains: $(ls $datadir | wc -l)"
echo "Total number of images: $(wc -l database_info.txt)"
echo "Sample of the database_info.txt file:"
head -n 10 database_info.txt
echo "..."
echo "-------------------"