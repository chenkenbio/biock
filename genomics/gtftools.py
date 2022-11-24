

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from .toolbox import parse_gtf_record
from ..utils import copen, run_bash

def gtf_to_bed(gtf, feature_type, attrs: List[str], sep: str='|', zero_start: bool=False):
    with copen(gtf) as infile:
        for l in infile:
            if l.startswith("#"):
                continue
            record = parse_gtf_record(l)
            if feature_type != "all" and record.feature_type != feature_type:
                continue
            name = list()
            for a in attrs:
                v = record.attrs.get(a, "NaN")
                if isinstance(v, list):
                    v = ','.join(v)
                name.append(v)
            name = sep.join(name)
            if zero_start:
                start, end = str(record.start), str(record.end)
            else:
                start, end = str(record.start - 1), str(record.end)
            print('\t'.join([
                record.chrom,
                start, 
                end,
                name,
                '.',
                record.strand
            ]))


def gtftogenepred(
        gtf: str, 
        ignore_groups_without_exons: bool=True,
        gene_pred_ext: bool=True, 
        gene_name_as_name2: bool=False,  
        include_version: bool=False):
    if gtf.endswith(".gtf.gz"):
        output = '.'.join(gtf.split('.')[:-2])
    elif gtf.endswith(".gtf"):
        output = '.'.join(gtf.split('.')[:-1])
    output = output + ".genepred.txt"
    cmd = ['gtfToGenePred']
    if ignore_groups_without_exons:
        cmd.append("-ignoreGroupsWithoutExons")
    if gene_pred_ext:
        cmd.append("-genePredExt")
    if gene_name_as_name2:
        cmd.append("-geneNameAsName2")
    if include_version:
        cmd.append("-includeVersion")
    cmd.append(gtf)
    cmd.append(output)
    cmd.append("&> {}.log".format(output))

    rc, out, err = run_bash(' '.join(cmd))

    if rc != 0:
        raise ValueError(out + "\n" + err + "\nsee {}.log".format(output))
    
    

