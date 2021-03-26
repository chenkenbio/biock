#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: $0 bed [strand]"
    exit 1
fi

bed="$1"
if [ -z $2 ]; then
    echo "Warning: strand ignored!" >&2
fi

fasta=$HOME/db/gencode/GRCh37/GRCh37.primary_assembly.genome.fa

bedtools getfasta -fi $fasta -bed $bed
