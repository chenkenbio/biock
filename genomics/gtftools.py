

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from .toolbox import parse_gtf_record
from ..utils import copen

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
