

from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from .toolbox import parse_gtf_record
from ..utils import copen

def gtf_to_bed(gtf, attrs: List[str], sep: str='|'):
    with copen(gtf) as infile:
        for l in infile:
            if l.startswith("#"):
                continue
            record = parse_gtf_record(l)
            name = list()
            for a in attrs:
                v = record.attrs.get(a, "NaN")
                name.append(v)
            name = sep.join(name)
            print('\t'.join([
                record.chrom,
                str(record.start - 1),
                str(record.end),
                name,
                '.',
                record.strand
            ]))
