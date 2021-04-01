#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage: bed [--no-chr]"
    exit 1
fi

bed="$1"
no_chr="$2"
if [ "$no_chr" = "--no-chr" ]; then
    sed 's/^chr//;s/^X/23/;s/^Y/24/;s/^MT/25/;s/^M/25/' $bed | sort -k1,1n -k2,2n -k3,3n | sed 's/^23\t/X\t/; s/^24\t/Y\t/; s/^25\t/M\t/'
else
    sed 's/^chr//;s/^X/23/;s/^Y/24/;s/^MT/25/;s/^M/25/' $bed | sort -k1,1n -k2,2n -k3,3n | sed 's/^23\t/X\t/; s/^24\t/Y\t/; s/^25\t/M\t/;s/^/chr/'
fi

