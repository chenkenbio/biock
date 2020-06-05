#!/bin/bash

delimiter='\t'
nlines=1

while [ -n "$1" ]; do
    if [ $1 = "-d" ]; then
        shift
        delimiter=$1
        shift
    elif [ $1 = "-n" ]; then
        shift
        nlines=$1
        shift
    elif [ -z $file ]; then
        file=$1
        shift
    fi
done

#if [ -z "$file" ] || [ ! -e $file ]; then
if [ -z "$file" ]; then
    echo "usage: index-head.sh file [-d delimiter] [-n lines]"
    echo "error: argument \"file\" is required"
    exit 1
fi

#echo "file:$file delimiter:$delimiter lines:$nlines"

# head -n $nlines $file | \

if [[ "$file" == *gz ]]; then
    zcat "$file" | head -n $nlines | \
        awk -F "$delimiter" '{for(i = 1; i <= NF; i++) if (i < NF) printf("%d:%s\t", i,$i); else printf("%d:%s\n", i, $i)}'
else
    cat "$file" | head -n $nlines | \
        awk -F "$delimiter" '{for(i = 1; i <= NF; i++) if (i < NF) printf("%d:%s\t", i,$i); else printf("%d:%s\n", i, $i)}'
fi
