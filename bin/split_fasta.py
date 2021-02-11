#!/usr/bin/env python3

with open("./GRCh37.primary_assembly.genome.fa") as infile:
    f = None
    for l in infile:
        if l.startswith('>'):
            header = l.lstrip('>').split(' ')[0].split('\t')[0]
            if f:
                f.close()
            f = open(header, 'w')
            f.write(l)
        else:
            f.write(l)
    f.close()
