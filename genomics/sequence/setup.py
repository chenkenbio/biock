#!/usr/bin/env python3
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="sequqnce_to_list",
    ext_modules=cythonize("./_sequence_to_kmer_list.pyx"),
    zip_safe=False,
)
