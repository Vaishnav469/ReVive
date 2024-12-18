#!/bin/bash

# Directory to process (default is the current directory)
DIR="${1:-.}"

# Initialize a total counter
total_files=0

# Loop through all directories and subdirectories
for folder in "$DIR"/*; do
    if [ -d "$folder" ]; then
        # Count files in this folder
        count=$(find "$folder" -type f | wc -l)
        echo "Folder: $folder -> Files: $count"
        total_files=$((total_files + count))
    fi
done

# Count files directly in the top-level directory
top_level_count=$(find "$DIR" -maxdepth 1 -type f | wc -l)
echo "Top-level directory -> Files: $top_level_count"

# Add top-level files to the total count
total_files=$((total_files + top_level_count))

# Display the grand total
echo "Total files in all folders: $total_files"

