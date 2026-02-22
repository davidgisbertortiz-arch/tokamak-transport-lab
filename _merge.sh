#!/bin/bash
set -e
cd /workspaces/tokamak-transport-lab

# If merge is in progress, complete it
if [ -f .git/MERGE_HEAD ]; then
    echo "Merge in progress — completing commit..."
    GIT_EDITOR=true git commit --no-edit
    echo "Merge commit done."
else
    echo "No merge in progress."
fi

echo "=== git log ==="
git log --oneline -5

echo "=== git status ==="
git status --short

echo "=== SCRIPT DONE ==="
