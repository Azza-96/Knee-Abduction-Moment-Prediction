#!/bin/bash

# -------------------------------------------------
# Smart Git Push Script with Ordered Push
# - LOCO folder → push whole
# - LOSO folder → push whole
# - Top-level files → push whole
# - Rest → size-based push (<2GB folder, >2GB subfolder/file)
# - Only ExtractedCSV uses Git LFS
# -------------------------------------------------

BRANCH=$(git branch --show-current || echo "main")
REMOTE_URL="https://github.com/Azza-96/Knee-Abduction-Moment-Prediction.git"
MAX_FOLDER_SIZE=$((2 * 1024 * 1024 * 1024))  # 2GB in bytes
LFS_FOLDER="ExtractedCSV"

# -------------------------------
# Initialize Git if needed
# -------------------------------
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git branch -M $BRANCH
    git remote add origin $REMOTE_URL
fi

# -------------------------------
# Setup Git LFS only for ExtractedCSV
# -------------------------------
git lfs install
git lfs track "$LFS_FOLDER/**/*.csv"
git add .gitattributes
git commit -m "Track ExtractedCSV CSV files with Git LFS" --allow-empty --no-verify
git push origin $BRANCH --force

# -------------------------------
# Helper functions
# -------------------------------
folder_size() {
    du -sb "$1" | awk '{print $1}'
}

push_folder() {
    local folder=$1
    echo "Pushing folder: $folder"
    git reset
    git add "$folder"
    git commit -m "Add/update folder $folder" --allow-empty --no-verify
    git push origin $BRANCH --force
}

push_files_individually() {
    local folder=$1
    FILES=$(find "$folder" -type f)
    for FILE in $FILES; do
        echo "Processing file: $FILE"
        git reset
        git add "$FILE"
        git commit --only "$FILE" -m "Add/update $FILE" --no-verify
        git push origin $BRANCH --force
    done
}

# -------------------------------
# 1. Push LOCO folder whole
# -------------------------------
if [ -d "LOCO" ]; then
    push_folder "LOCO"
fi

# -------------------------------
# 2. Push LOSO folder whole
# -------------------------------
if [ -d "LOSO" ]; then
    push_folder "LOSO"
fi

# -------------------------------
# 3. Push top-level files (not folders) whole
# -------------------------------
TOP_FILES=$(find . -maxdepth 1 -type f)
if [ ! -z "$TOP_FILES" ]; then
    echo "Pushing top-level files"
    git reset
    git add $TOP_FILES
    git commit -m "Add/update top-level files" --allow-empty --no-verify
    git push origin $BRANCH --force
fi

# -------------------------------
# 4. Apply size-based push for the rest
# -------------------------------
# Ignore LOCO, LOSO, top-level files, and ExtractedCSV handled via LFS
IGNORE_LIST="LOCO LOSO $LFS_FOLDER"
TOP_FOLDERS=$(find . -mindepth 1 -maxdepth 1 -type d)

for FOLDER in $TOP_FOLDERS; do
    BASENAME=$(basename "$FOLDER")
    if [[ " $IGNORE_LIST " =~ " $BASENAME " ]]; then
        continue
    fi
    SIZE=$(folder_size "$FOLDER")
    if [ "$SIZE" -le "$MAX_FOLDER_SIZE" ]; then
        push_folder "$FOLDER"
    else
        # Folder too big → check subfolders
        SUBFOLDERS=$(find "$FOLDER" -mindepth 1 -maxdepth 1 -type d)
        for SUB in $SUBFOLDERS; do
            SUBSIZE=$(folder_size "$SUB")
            if [ "$SUBSIZE" -le "$MAX_FOLDER_SIZE" ]; then
                push_folder "$SUB"
            else
                push_files_individually "$SUB"
            fi
        done
        # Also push loose files in top-level folder individually
        FILES=$(find "$FOLDER" -maxdepth 1 -type f)
        for FILE in $FILES; do
            git reset
            git add "$FILE"
            git commit --only "$FILE" -m "Add/update $FILE" --no-verify
            git push origin $BRANCH --force
        done
    fi
done

# -------------------------------
# 5. Push ExtractedCSV via LFS
# -------------------------------
if [ -d "$LFS_FOLDER" ]; then
    push_folder "$LFS_FOLDER"
fi

echo "----------------------------------------"
echo "All files/folders pushed successfully. LOCO, LOSO, top-level files, and ExtractedCSV handled appropriately."
