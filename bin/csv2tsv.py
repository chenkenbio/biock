#!/usr/bin/env python3
"""Convert csv file to tsv file"""

import os, csv, argparse

def get_args():
    parser = argparse.ArgumentParser(description=\
        'Convert csv file to tsv(or other delimiter) file')
    parser.add_argument('csv', nargs='+', type=str, help='input csv file(s)')
    parser.add_argument('-d', dest='delimiter', type=str, default='\t',\
        required=False, help='output delimiter, default is "\\t"')
    parser.add_argument('-o',dest="to_file",action="store_true",\
        required=False, help="writting output to file directly")
    parser.add_argument('-w', action='store_true', help="")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    for f in args.csv:
        if not os.path.isfile(f):
            exit("ERROR: \'" + f + "\' is not regular file")
    
    for f in args.csv:
        with open(f) as infile:
            if args.to_file:
                with open(f.rstrip('.csv') + '.tsv', 'w') as outfile:
                    csvin = csv.reader(infile)
                    for l in csvin:
                        outfile.write(args.delimiter.join(l) + '\n')
            else:
                csvin = csv.reader(infile)
                for l in csvin:
                    print(args.delimiter.join(l))
