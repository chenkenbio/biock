#!/usr/bin/env python3

import argparse
import os
import time
import sys
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
from biock import make_directory, make_logger, backup_file, check_exists
from biock.pytorch import set_seed

TO_SAVE = {
    __file__
}


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


    ## common args
    p.add_argument('-o', "--outdir", required=True)
    p.add_argument("--lr", type=float, default=1E-3, help="learning rate")
    p.add_argument("--batch-size", type=int, default=4, help="batch size")
    p.add_argument("--epoch-num", type=int, default=500)
    p.add_argument("--num-workers", type=int, default=0)

    task = p.add_mutually_exclusive_group()
    task.add_argument("--pretrain", type=str, help="run model using pre-trained model")
    task.add_argument("--resume", type=str, help="resume training")
    task.add_argument("--transfer", type=str, help="transfer learning using")

    p.add_argument('--seed', type=int, default=2020)
    return p

if __name__ == "__main__":
    args = get_args().parse_args()
    set_seed(args.seed)

    ## check args
    check_exists(args.pretrain, args.resume, args.transfer)

    ## setup 
    outdir = make_directory(args.outdir)
    logger = make_logger(title=__name__, filename="{}/run.log".format(outdir), trace=True)
    logger.info("##{}".format(time.time()))

    src_dir = make_directory("{}/src".format(outdir))
    for m in TO_SAVE:
        dst = backup_file(m, src_dir, readonly=True)
        logger.warning("- backup {}  => {}".format(m, dst))

    logger.info("##cmd: {}".format(' '.join(sys.argv)))
    logger.info("##pwd: {}".format(os.getcwd()))
    logger.info("##args: {}\n".format(args))

    ## load data


    ## build model
    model_config = {

    }
    model: nn.Module=None

    ## training/transfer learning/resume training
    if args.pretrain:
        d = torch.load(args.pretrain)
        model.load_state_dict(

        )
    else:
        model.fit(

            outdir=outdir,
            batch_size=args.batch_size,
            lr=args.lr,
            epoch_num=args.epoch_num,
            resume=args.resume,
            transfer=args.transfer,
            seed=args.seed
        )
    

    ## evaluation





