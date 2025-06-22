#!/bin/bash

# ==============================================================================
# MkDocs Mathematics Structure Scaffolding Script (v2)
#
# This script will scaffold the domain-based Mathematics structure
# within the './docs/math/' directory.
# Now includes "Combinatorial Optimization".
# ==============================================================================

echo "--- Scaffolding new Mathematics structure (v2)... ---"

# Define the new Math structure.
# Format: "path/relative/to/docs;Page Title"
math_structure=$(cat <<- 'EOF'
math/index.md;数学概述与思想
math/analysis/mathematical_analysis.md;数学分析 III
math/analysis/real_analysis.md;实变函数
math/analysis/complex_analysis.md;复变函数
math/analysis/ode.md;常微分方程
math/analysis/pde.md;偏微分方程
math/algebra_geometry/higher_algebra.md;高等代数与解析几何
math/algebra_geometry/abstract_algebra.md;抽象代数
math/algebra_geometry/topology.md;点集拓扑
math/prob_stats/probability_theory.md;概率论
math/prob_stats/mathematical_statistics.md;数理统计
math/prob_stats/data_modeling.md;数据建模与分析
math/applied/optimization_algorithms.md;优化实用算法
math/applied/combinatorial_optimization.md;组合优化 # <--- 新增行
math/history.md;数学史
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

done <<< "$math_structure"

echo ""
echo "✅ Mathematics framework (v2) has been created successfully."
echo "Next step: Please manually update the 'Mathematics' section in your mkdocs.yml's nav."