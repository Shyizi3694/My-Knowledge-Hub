#!/bin/bash

# ==============================================================================
# MkDocs CS Structure Refactoring Script
#
# This script will:
# 1. Ask for confirmation.
# 2. Clean up ONLY the './docs/cs/' directory and the './docs/demo.md' file.
# 3. Scaffold the new domain-based Computer Science structure within './docs/cs/'.
# ==============================================================================


# --- Phase 1: Safety Confirmation ---
# ------------------------------------------------------------------------------
echo "⚠️ WARNING: This script will perform a targeted refactoring."
echo "It will permanently delete the following specific items:"
echo "  - The entire './docs/cs/' directory and all its contents."
echo "  - The './docs/demo.md' file."
echo "Other directories like 'skills/', 'math/', etc., will NOT be affected."
echo ""
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo # Move to a new line
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled by user."
    exit 1
fi


# --- Phase 2: Targeted Cleanup ---
# ------------------------------------------------------------------------------
echo ""
echo "--- Cleaning up old CS structure and demo file... ---"

# List of specific items to delete
targets_to_delete=(
    "docs/cs"
)

for target in "${targets_to_delete[@]}"; do
    if [ -e "$target" ]; then # Check if file or directory exists
        echo "Deleting $target..."
        rm -rf "$target"
    else
        echo "Skipping $target (not found)."
    fi
done

echo "Cleanup complete."
echo ""


# --- Phase 3: Scaffold New Computer Science Structure ---
# ------------------------------------------------------------------------------
echo "--- Scaffolding new Computer Science structure... ---"

# Define the new CS structure.
cs_structure=$(cat <<- 'EOF'
cs/index.md;计算机科学导论与概述
cs/theory/discrete_math.md;离散数学及其应用
cs/theory/data_structures_algorithms.md;数据结构与算法分析
cs/theory/numerical_analysis.md;数值分析
cs/languages/c_basics.md;C 程序设计基础
cs/languages/oop_cpp.md;面向对象程序设计 (C++)
cs/systems/digital_logic.md;数字逻辑设计
cs/systems/organization.md;计算机组成
cs/systems/architecture.md;计算机体系结构
cs/ai/intro_to_ai.md;人工智能导论
cs/ai/data_mining.md;数据挖掘导论
cs/ai/image_processing.md;图像信息处理
cs/applied/database_systems.md;数据库系统
cs/applied/security_principles.md;信息安全原理
EOF
)

# Main docs directory
DOCS_DIR="docs"

# Read the structure line by line and create files
while IFS= read -r line; do
    if [ -z "$line" ]; then continue; fi

    filepath=$(echo "$line" | cut -d';' -f1)
    title=$(echo "$line" | cut -d';' -f2)
    full_path="$DOCS_DIR/$filepath"

    mkdir -p "$(dirname "$full_path")"
    echo "# $title" > "$full_path"
    echo "Created: $full_path"

done <<< "$cs_structure"

echo ""
echo "✅ Refactoring complete!"
echo "The 'cs' directory has been rebuilt with the new structure."
echo "Next step for you: Please manually update the 'Computer Science' section in your mkdocs.yml's nav."