#!/usr/bin/env python3
"""
Author: Ken Chen (chenkenbio@gmail.com)
Date: <<date>>
"""

import argparse
import os
import io
import sys 
import json
from subprocess import Popen, PIPE

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('id')
    p.add_argument("-t", "--type", default="files")
    p.add_argument("-f", "--force", action="store_true")
    p.add_argument("-o", "--output")
    p.add_argument('--seed', type=int, default=2020)
    return p


if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()

    if args.output is None:
        args.output = args.id + ".json"

    if os.path.exists(args.output) and not args.force:
        print("{} exists.".format(args.output))
    else:
        p = Popen(['/bin/bash', '-c', 'curl -L -H "Accept: application/json" https://www.encodeproject.org/{}/{}/'.format(args.type, args.id)], stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(err.decode("utf8"))
        out = json.load(io.StringIO(out.decode("utf8")))
        json.dump(out, open(args.output, 'w'), indent=4)

