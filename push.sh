#!/bin/bash

# --- Configuration ---
TARGET_FOLDER="ExtractDataForML_KAM_Prediction/V3D Exported Data"
BRANCH=$(git branch --show-current || echo "main")

echo "Starting file-by-file push for folder: $TARGET_FOLDER"
echo "Target Branch: $BRANCH"

# Check if the folder exists
if [ ! -d "$TARGET_FOLDER" ]; then
    echo "Error: Folder '$TARGET_FOLDER' does not exist."
    exit 1
fi

# Find all files within the target folder and loop through them
find "$TARGET_FOLDER" -type f | while read -r FILE; do
    echo "Processing file: $FILE"
    
    # Check if the file is tracked by Git (i.e., not a new file)
    if git ls-files --error-unmatch "$FILE" >/dev/null 2>&1; then
        # Check if the file has local modifications
        if ! git diff --exit-code "$FILE" >/dev/null 2>&1; then
            echo "-> File modified. Committing and pushing update."
            # Reset the index to ensure only this file is staged
            git reset
            git add "$FILE"
            git commit --only "$FILE" -m "Update $FILE" --no-verify
            git push origin "$BRANCH"
        else
            echo "-> File exists and is unchanged. Skipping."
        fi
    else
        echo "-> New file found. Committing and pushing."
        # Reset the index to ensure only this file is staged
        git reset
        git add "$FILE"
        git commit --only "$FILE" -m "Add $FILE" --no-verify
        git push origin "$BRANCH"
    fi
done

echo "---"
echo "File-by-file push completed for $TARGET_FOLDER."

# Final cleanup of any potential pending changes (like the smart_push.sh or this script itself)
# Note: Since you explicitly don't want smart_push.sh committed, we use restore to keep it local.
if [ -f "smart_push.sh" ] && ! git diff --exit-code smart_push.sh >/dev/null 2>&1; then
    echo "Restoring smart_push.sh modifications."
    git restore smart_push.sh
fi

echo "Operation complete. Check remote repository for files."
