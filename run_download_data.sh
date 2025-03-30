#!/bin/bash

# Dataset URL and save path
DATA_URL="https://techassessment.blob.core.windows.net/aiap-pys-2/noshow.db"
SAVE_PATH="data/noshow.db"

# Check if the dataset already exists
if [ -f "$SAVE_PATH" ]; then
    echo "📄 $SAVE_PATH already exists, skipping download."
    exit 0
fi

# Create the data directory if it doesn't exist
echo "📂 Creating directory for $SAVE_PATH if it doesn't exist..."
mkdir -p "$(dirname "$SAVE_PATH")"

# Download the dataset
echo "⏬ Downloading from $DATA_URL..."
if curl -o "$SAVE_PATH" "$DATA_URL" --fail --silent --show-error; then
    echo "💾 Saved to $SAVE_PATH"
    echo "✅ Download complete!"
else
    echo "❌ Error: Download failed."
    exit 1
fi