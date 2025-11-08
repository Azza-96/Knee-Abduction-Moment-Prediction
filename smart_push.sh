#!/bin/bash

# -------------------------------------------------
# Smart Git Push Script with Ordered Push
# - LOCO folder → push whole
# - LOSO folder → push whole
# - Top-level files → push whole
# - Rest → size-based push (<2GB folder, >2GB subfolder/file)
# - Only ExtractedCSV uses Git LFS
# - Exclude .git and .gitattributes from general loops
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
    FILES=$(find "$folder" -type f ! -path "*/.git/*")
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
[ -d "LOCO" ] && push_folder "LOCO"

# -------------------------------
# 2. Push LOSO folder whole
# -------------------------------
[ -d "LOSO" ] && push_folder "LOSO"

# -------------------------------
# 3. Push top-level files (not folders) whole
# -------------------------------
TOP_FILES=$(find . -maxdepth 1 -type f ! -name ".gitattributes" ! -path "./.git/*")
[ ! -z "$TOP_FILES" ] && git add $TOP_FILES && git commit -m "Add/update top-level files" --allow-empty --no-verify && git push origin $BRANCH --force

# -------------------------------
# 4. Apply size-based push for the rest
# -------------------------------
IGNORE_LIST="LOCO LOSO $LFS_FOLDER"
TOP_FOLDERS=$(find . -mindepth 1 -maxdepth 1 -type d ! -name ".git")

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
        SUBFOLDERS=$(find "$FOLDER" -mindepth 1 -maxdepth 1 -type d ! -name ".git")
        for SUB in $SUBFOLDERS; do
            SUBSIZE=$(folder_size "$SUB")
            if [ "$SUBSIZE" -le "$MAX_FOLDER_SIZE" ]; then
                push_folder "$SUB"
            else
                push_files_individually "$SUB"
            fi
        done
        # Also push loose files in top-level folder individually
        FILES=$(find "$FOLDER" -maxdepth 1 -type f ! -path "*/.git/*")
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
[ -d "$LFS_FOLDER" ] && push_folder "$LFS_FOLDER"

echo "----------------------------------------"
echo "All files/folders pushed successfully. LOCO, LOSO, top-level files, and ExtractedCSV handled appropriately."
