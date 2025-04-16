#!/bin/bash

# Dataset URL and save path
DATA_URL="https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db"
SAVE_PATH="data/noshow.db"

# Check if the dataset exists
if [ ! -f "$SAVE_PATH" ]; then
    echo "❌ Error: Dataset '$SAVE_PATH' not found in the 'data' folder."
    
    # Prompt the user to download the dataset automatically
    read -p "   Do you want to download the dataset automatically? (y/n): " choice
    
    # Convert the input to lowercase for case-insensitive comparison
    choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')
    
    if [ "$choice" == "y" ] || [ "$choice" == "yes" ]; then
        echo "   └── Creating directory for $SAVE_PATH if it doesn't exist..."
        mkdir -p "$(dirname "$SAVE_PATH")"

        echo "   └── Downloading from $DATA_URL..."
        if curl -o "$SAVE_PATH" "$DATA_URL" --fail --silent --show-error; then
            echo "   └── Saving to $SAVE_PATH..."
            echo "✅ Dataset downloaded successfully!"
            echo
        else
            echo "❌ Error: Download failed."
            exit 1
        fi
    elif [ "$choice" == "n" ] || [ "$choice" == "no" ]; then
        echo "   Exiting pipeline. Please manually place the dataset in the 'data' folder and try again."
        exit 1
    else
        echo "❌ Invalid input. Please run the command again and enter 'y' or 'n'."
        echo "   Exiting pipeline."
        exit 1
    fi
fi

# Run the machine learning pipeline with configurable parameters
echo "🚀🚀🚀 Running the machine learning pipeline..."
echo
python src/pipeline.py

# Check if the pipeline executed successfully
if [ $? -eq 0 ]; then
    echo "🍻 Pipeline executed successfully!"
else
    echo
    echo "❌ Error: Pipeline execution failed."
    exit 1
fi