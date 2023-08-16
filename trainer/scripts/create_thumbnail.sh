#!/bin/bash

# Source and destination directories
source_dir="/media/saplab/MinhNVMe/relahash/data/shopee_17-8/product_images"
dest_dir="/media/saplab/MinhNVMe/relahash/data/shopee_17-8/thumbnail"

# Create the destination directory if it doesn't exist
mkdir -p "$dest_dir"

# Function to process images recursively
process_images() {
    for file in "$1"/*; do
        if [ -d "$file" ]; then
            # If the item is a directory, recursively process it
            sub_dir="$dest_dir/${file#$source_dir/}"
            mkdir -p "$sub_dir"
            process_images "$file"
        elif [ -f "$file" ]; then
            # If the item is a file, process the image
            sub_dir="$dest_dir/${file#$source_dir/}"
            output_file="$sub_dir/${file##*/}"
            
            echo "Processing $file -> $output_file"
            echo convert "$file" -resize x150 "$output_file"
        fi
    done
}

# Start processing from the source directory
process_images "$source_dir"

echo "Image processing complete."
