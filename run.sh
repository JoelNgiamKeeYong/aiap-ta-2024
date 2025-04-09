#!/bin/bash

# Check if the dataset exists
if [ ! -f "data/noshow.db" ]; then
    echo "❌ Error: Dataset 'data/noshow.db' not found. Please ensure the dataset is placed in the 'data' folder"
    exit 1
fi

# Run the machine learning pipeline with configurable parameters
echo "🚀 Running the machine learning pipeline... 🛠️"
python src/pipeline.py

# Check if the pipeline executed successfully
if [ $? -eq 0 ]; then
    echo "✅ Pipeline executed successfully!"
else
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi