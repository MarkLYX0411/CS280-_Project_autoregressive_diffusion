#!/bin/bash

# List of categories to download
categories=(
    "apple"
    "cat"
    "cloud"
    "computer"
    "diamond"
    "eyeglasses"
    "moon"
    "mushroom"
    "snowflake"
    "star"
)

# Base URL for QuickDraw dataset
base_url="https://storage.googleapis.com/quickdraw_dataset/full/raw"

# Number of drawings to download for each category
num_drawings=1500

for category in "${categories[@]}"; do
  echo "Downloading first $num_drawings drawings of $category.ndjson..."
  # Download the first part of the file
  curl -o "${category}_temp.ndjson" "$base_url/$category.ndjson" --range 0-10485760
  
  # Extract the first N complete drawings
  head -n $num_drawings "${category}_temp.ndjson" > "${category}.ndjson"
  
  # Clean up
  rm "${category}_temp.ndjson"
done

echo "All partial downloads complete!"