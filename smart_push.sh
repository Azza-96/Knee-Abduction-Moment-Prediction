#!/bin/bash

# -------------------------------------------------
# Robust Smart Git Push Script
# -------------------------------------------------

BRANCH=$(git branch --show-current || echo "main")
REMOTE_URL="https://github.com/Azza-96/Knee-Abduction-Moment-Prediction.git"
MAX_FOLDER_SIZE=$((2 * 1024 * 1024 * 1024)) # 2GB
LFS_FOLDER="ExtractedCSV"

# Initialize git if needed 
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
    git branch -M "$BRANCH"
    git remote add origin "$REMOTE_URL"
fi

# Setup Git LFS for ExtractedCSV
git lfs install
# Assuming this command is already correctly tracked, but running again is safe
git lfs track "$LFS_FOLDER/**/*.csv"
git add .gitattributes
git commit -m "Track ExtractedCSV CSV files with Git LFS" --allow-empty --no-verify

# Helper functions
# Uses 'du -sb' (disk usage in size bytes) to get folder size
folder_size() {
    # Check if the folder exists and is a directory
    if [ -d "$1" ]; then
        du -sb "$1" 2>/dev/null | awk '{print $1}'
    fi
}

# Pushes a folder as one commit (for small folders)
push_folder() {
    local folder="$1"
    [ ! -d "$folder" ] && return
    echo "Pushing folder: $folder"
    git add "$folder"
    git commit -m "Add/update folder $folder" --allow-empty --no-verify
    git push origin "$BRANCH"
}

# Pushes files individually (for large folders/subfolders)
push_files_individually() {
    local folder="$1"
    [ ! -d "$folder" ] && return
    find "$folder" -type f | while read -r FILE; do
        echo "Processing file: $FILE"
        git reset
        git add "$FILE"
        git commit --only "$FILE" -m "Add/update $FILE" --no-verify
        git push origin "$BRANCH"
    done
}

# Push LOCO and LOSO whole
for FOLDER in LOCO LOSO; do
    push_folder "./$FOLDER"
done

# Push top-level files (not folders)
find . -maxdepth 1 -type f ! -name "*.sh" ! -name ".gitattributes" | while read -r FILE; do
    echo "Adding top-level file: $FILE"
    git reset
    git add "$FILE"
    git commit --only "$FILE" -m "Add/update top-level file $FILE" --no-verify
    git push origin "$BRANCH"
done

# Push ExtractedCSV via LFS, file by file
push_files_individually "$LFS_FOLDER"

# Push remaining folders using 2GB logic
# This list now includes the specific parent folder we need to check
TARGET_FOLDERS="./ProcessedCyclesPrediction ./ProcessedCycles ./Segement_cycles_Save_indice ./statPeak ./ExtractDataForML_KAM_Prediction"

for FOLDER_PATH in $TARGET_FOLDERS; do
    
    # Use find to properly resolve paths and skip non-existent folders
    FOLDER_PATH=$(echo "$FOLDER_PATH" | xargs -n1)
    [ -z "$FOLDER_PATH" ] && continue
    [ ! -d "$FOLDER_PATH" ] && continue
    
    SIZE=$(folder_size "$FOLDER_PATH")
    
    # Check if the size calculation was successful and is a number
    if [ -n "$SIZE" ] && [[ "$SIZE" =~ ^[0-9]+$ ]]; then
        
        # Check if a top-level folder is too big
        if [ "$SIZE" -le "$MAX_FOLDER_SIZE" ]; then
            push_folder "$FOLDER_PATH"
        else
            echo "Folder $FOLDER_PATH exceeds size limit ($MAX_FOLDER_SIZE bytes). Processing contents individually."
            
            # Process subfolders within the big folder
            # This is where 'V3D Exported Data' will be processed
            SUBS=$(find "$FOLDER_PATH" -mindepth 1 -maxdepth 1 -type d)
            for SUB in $SUBS; do
                SUBSIZE=$(folder_size "$SUB")
                
                # CRITICAL FIX: Robust check for subfolder size before comparison
                if [ -n "$SUBSIZE" ] && [[ "$SUBSIZE" =~ ^[0-9]+$ ]]; then
                    if [ "$SUBSIZE" -le "$MAX_FOLDER_SIZE" ]; then
                        push_folder "$SUB"
                    else
                        echo "Subfolder $SUB exceeds size limit. Processing files individually."
                        push_files_individually "$SUB"
                    fi
                else
                    # Fallback if size calculation failed (e.g., V3D Exported Data/ is empty)
                    echo "Warning: Could not determine size for subfolder $SUB. Pushing as single folder."
                    push_folder "$SUB" 
                fi
            done
            
            # Handle loose files in the top-level folder (if any)
            echo "Processing loose files in $FOLDER_PATH..."
            find "$FOLDER_PATH" -maxdepth 1 -type f | while read -r FILE; do
                git reset
                git add "$FILE"
                git commit --only "$FILE" -m "Add/update $FILE" --no-verify
                git push origin "$BRANCH"
            done
        fi
    else
        echo "Warning: Could not determine size for folder $FOLDER_PATH. Pushing as single folder."
        push_folder "$FOLDER_PATH"
    fi
done

echo "All files/folders pushed. LOCO, LOSO, top-level files, ExtractedCSV (LFS) and remaining folders handled."
