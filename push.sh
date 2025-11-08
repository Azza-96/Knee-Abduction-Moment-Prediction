#!/bin/bash

FOLDER="ExtractedCSV"
BRANCH=$(git branch --show-current)

# Ensure we are inside a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not a git repository."
    exit 1
fi

# Unstage everything to avoid including other changes
git reset

# List all files in folder
FILES=$(find "$FOLDER" -type f)

if [ -z "$FILES" ]; then
    echo "No files found in $FOLDER."
    exit 0
fi

echo "Found $(echo "$FILES" | wc -l) files in $FOLDER."

for FILE in $FILES; do
    echo "----------------------------------------"
    echo "Processing: $FILE"

    # Stage only this file
    git add "$FILE"

    # Commit only this file explicitly
    git commit --only "$FILE" -m "Add or update $FILE" --no-verify

    # Push this single commit
    git push origin "$BRANCH"

    # Unstage everything again to ensure next commit is clean
    git reset
done

echo "----------------------------------------"
echo "Done! Each file in $FOLDER has been committed and pushed individually."
