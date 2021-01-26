#!/bin/bash
# Move files in the form DIR/TIER_* into subdirs, to form a DIR/TIER/TIER_* directory structure.
DIR="$*"
ARR=()
for f in "$DIR"/*; do
    [ -d "$f" ] && continue

    TIER="${f%%_*}"
    if [ ! -d "./$TIER" ]; then
        mkdir -v "./$TIER"
        ARR+=("./$TIER")
        continue
    fi

    MATCH=0
    for d in "${ARR[@]}"; do
        if [ "$d" == "./$TIER" ]; then
            MATCH=1
            break
        fi
    done

    if [ "$MATCH" == 0 ]; then
        ARR+=("./$TIER")
    fi
done

for d in "${ARR[@]}"; do
    echo "moving files to $d"
    mv -vn "./${d}_"* "./${d}"
done
