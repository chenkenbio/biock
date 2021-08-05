#!/usr/bin/env python3

import os, json
import logging

NCBI_GENE_NAME: dict = json.load(open(os.path.join(os.path.dirname(__file__), "NCBI_gene2symbol.json")))
ncbi_gene_name = NCBI_GENE_NAME

NCBI_GENE_ID: dict = json.load(open(os.path.join(os.path.dirname(__file__), "NCBI_gene2id.json")))
NCBI_GENE_ENTREZ = NCBI_GENE_ID

def to_std_gene_name(name, report=False):
    # status: (True, ambiguous, missing)
    if name in NCBI_GENE_NAME:
        name_ = NCBI_GENE_NAME[name]
        if ';' in name_:
            logging.warning("Ambiguous standard gene names: {} -> {}, unchanged ({})".format(name, name_, name))
            status = "ambiguous"
        else:
            name = name_
            status = True
    else:
        logging.warning("Not found in standard gene names: {}".format(name))
        status = "missing"

    if report:
        return name, status
    else:
        return name
