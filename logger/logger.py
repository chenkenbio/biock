import os
import shutil
import time
import logging
from typing import Literal, Optional
import warnings

def make_logger(
        title: Optional[str]="", 
        filename: Optional[str]=None, 
        level: Literal["INFO", "DEBUG"]="INFO", 
        filemode: Literal['w', 'a']='w',
        show_line: bool=False):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if show_line:
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            shutil.move(filename, "{}.{}".format(filename, suffix))
            warnings.warn("log {} exists, moved to to {}.{}".format(filename, filename, suffix))
        fh = logging.FileHandler(filename=filename, mode=filemode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

