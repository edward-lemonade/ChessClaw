#!/bin/bash
#
# run this in git bash, under the project root
# ./dataset/prepare_dataset.sh

cat processed.csv |
    tr -d '"' |
    tr '\' '/' |
    while IFS=, read -r file name; do
        label="${name/ /_}"
        SN="$(basename $file | cut -d'_' -f2)"
        if [[ "$SN" -ge 512 ]]; then
            dests=( test )
        else
            dests=( train validate )
        fi
        for d in "${dests[@]}"; do
            folder="dataset/chessv/$d/$label"
            mkdir -p "$folder"
            cp "$file" "$folder"
        done
    done