#!/bin/bash

# Reset the project by deleting data, models, and output folders
echo "🔄 Resetting the project..."

# Define directories to delete
DIRECTORIES=("data" "models" "output")

# Loop through each directory and delete if it exists
for DIR in "${DIRECTORIES[@]}"; do
    if [ -d "$DIR" ]; then
        echo "   └── Deleting '$DIR' folder..."
        rm -rf "$DIR"
    else
        echo "   └── '$DIR' folder does not exist. Skipping deletion."
    fi
done

# Check if the /archives/training_logs.txt file exists and ask for confirmation
LOG_FILE="archives/training_logs.txt"
if [ -f "$LOG_FILE" ]; then
    echo
    read -p "❓  Do you want to delete the '$LOG_FILE' file? (y/n): " CONFIRMATION
    if [[ "$CONFIRMATION" == "y" || "$CONFIRMATION" == "Y" ]]; then
        echo "   └── Deleting '$LOG_FILE' file..."
        rm -f "$LOG_FILE"
    else
        echo "   └── Skipping deletion of '$LOG_FILE'."
    fi
fi

# Final confirmation
echo
echo "✅ Project reset completed successfully!"