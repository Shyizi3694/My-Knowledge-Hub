#!/bin/bash

# ==============================================================================
# MkDocs General Education Structure Scaffolding Script
#
# This script will scaffold the General Education courses structure
# within the './docs/gen_ed/' directory.
# It will not delete any existing files.
# ==============================================================================

echo "--- Scaffolding new General Education structure... ---"

# Define the new Gen-Ed structure.
structure=$(cat <<- 'EOF'
gen_ed/index.md;通识课程概述
gen_ed/natural_science/calculus_1.md;微积分 I
gen_ed/natural_science/calculus_2.md;微积分 II
gen_ed/natural_science/linear_algebra.md;线性代数
gen_ed/natural_science/physics_1.md;大学物理 I
gen_ed/natural_science/physics_2.md;大学物理 II
gen_ed/natural_science/physics_lab.md;大学物理实验
gen_ed/politics_military/ethics_law.md;思想道德与法治
gen_ed/politics_military/modern_history.md;中国近代史纲要
gen_ed/politics_military/military_theory.md;军事理论
gen_ed/politics_military/cpc_history.md;中国共产党史
gen_ed/politics_military/marxism_basics.md;马克思主义基本原理
gen_ed/politics_military/mao_thought.md;毛泽东思想概论
gen_ed/politics_military/xi_thought.md;习近平新时代中国特色社会主义思想概论
gen_ed/languages/english_3.md;大学英语 III
gen_ed/languages/english_4.md;大学英语 IV
gen_ed/electives/tea_culture.md;茶文化与茶健康
gen_ed/electives/scientific_instruments.md;现代科学仪器
gen_ed/electives/law_basics.md;法学基础
gen_ed/electives/economic_law.md;经济法理论与实务
gen_ed/electives/blockchain_practice.md;区块链技术应用实践
gen_ed/electives/calligraphy_history.md;中国书法史
gen_ed/career/career_planning.md;职业生涯规划
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

done <<< "$structure"

echo ""
echo "✅ General Education framework has been created successfully."
echo "Next step: Please manually update the 'General Education' section in your mkdocs.yml's nav."