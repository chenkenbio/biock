#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import argparse
from string import Formatter
from biock import auto_open
from typing import Iterable

def get_formatted_metadata(metadata: str, query_string: str) -> Iterable[str]:
    keywords = [x[1] for x in Formatter().parse(query_string)]
    results = list()
    with auto_open(metadata) as infile:
        for nr, l in enumerate(infile):
            fields = l.replace(' ','_').strip('\n').split('\t')
            if nr == 0:
                keys = fields.copy()
                key2col = {k:i for i, k in enumerate(fields)}
                missing = list()
                for i, k in enumerate(keywords):
                    if k not in key2col:
                        try:
                            k = int(k)
                            if k == 0 or k > len(keys):
                                raise ValueError("")
                            keywords[i] = int(k)
                        except ValueError as err:
                            missing.append(k)
                if len(missing) > 0:
                    raise KeyError("missing key(s): {}".format(', '.join(missing)))
            else:
                meta = dict()# {k:fields[key2col[k]] for k in keywords}
                for k in keywords:
                    if isinstance(k, int):
                        meta[str(k)] = fields[k]
                    else:
                        meta[k] = fields[key2col[k]]
                yield query_string.format(**meta)

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('metadata', help="metadata.tsv")
    p.add_argument('-f', "--format", required=True, help="string format")
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()

    for l in get_formatted_metadata(args.metadata, args.format):
        print(l)

