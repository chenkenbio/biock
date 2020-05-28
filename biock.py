#!/usr/bin/env python3

import argparse, os, sys, warnings, time, json, gzip, logging
import numpy as np
import subprocess
from subprocess import Popen, PIPE
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, precision_recall_curve, auc
import functools

print = functools.partial(print, flush=True)

### misc
def str2num(s):
    """ ideas from HuangBi """
    try:
        n = int(s)
    except:
        n = float(s)
    return n

def overlap(x1, x2, y1, y2):
    """ [x1, x2), [y1, y2) """
    length = 0
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if x2 <= y1:
        length = x2 - y1
    elif x1 <= y2:
        length = min(x2, y2) - max(x1, y1)
    else:
        length = y2 - x1
    return length



### logs
def print_run_info(args=None):
    print("\n# PROG: '{}' started at {}".format(os.path.basename(sys.argv[0]), time.asctime()))
    print("## PWD: %s" % os.getcwd())
    print("## CMD: %s" % ' '.join(sys.argv))
    if args is not None:
        print("## ARG: {}".format(args))

def prog_header(args=None, out=sys.stdout):
    print("\n# Started at {}".format(time.asctime()))
    print("## Command: {}".format(' '.join(sys.argv)), file=out)
    if args is not None:
        print("##: Args: {}".format(args))

def get_logger(log_level="INFO"):
    logging.basicConfig(
                format='[%(asctime)s %(levelname)s] %(message)s',
                    stream=sys.stdout
            )
    log = logging.getLogger(__name__)
    log.setLevel(log_level)
    return log


### file & directory
def dirname(fn):
    folder = os.path.basename(fn)
    if folder == '':
        folder = "./"
    return folder

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def run_bash(cmd):
    p = Popen(['/bin/bash', '-c', cmd], stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out, err = out.decode('utf8'), err.decode('utf8')
    rc = p.returncode
    return (rc, out, err)


## deep learning related
#def model_summary(model):
#    print("model_summary")
#    print("Layer_name"+"\t"*7+"Number of Parameters")
#    print("="*100)
#    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
#    layer_name = [child for child in model.children()]
#    j = 0
#    total_params = 0
#    print("\t"*10)
#    for i in layer_name:
#        print()
#        param = 0
#        try:
#            bias = (i.bias is not None)
#        except:
#            bias = False  
#        if not bias:
#            param =model_parameters[j].numel()+model_parameters[j+1].numel()
#            j = j+2
#        else:
#            param =model_parameters[j].numel()
#            j = j+1
#        print(str(i)+"\t"*3+str(param))
#        total_params+=param
#    print("="*100)
#    print(f"Total Params:{total_params}")     

def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


## evaluation
def aupr_score(true, prob):
    true, prob = np.array(true), np.array(prob)
    assert len(true.shape) == 1 or min(true.shape) == 1
    assert len(prob.shape) == 1 or min(prob.shape) == 1
    true, prob = true.reshape(-1), prob.reshape(-1)
    precision, recall, thresholds = precision_recall_curve(true, prob)
    aupr = auc(recall, precision)
    return aupr


## constants & variables
hg19_chromsize = {"chr1": 249250621, "chr10": 135534747, "chr11": 135006516, 
        "chr12": 133851895, "chr13": 115169878, "chr14": 107349540, 
        "chr15": 102531392, "chr16": 90354753, "chr17": 81195210, 
        "chr18": 78077248, "chr19": 59128983, "chr2": 243199373, 
        "chr20": 63025520, "chr21": 48129895, "chr22": 51304566, 
        "chr3": 198022430, "chr4": 191154276, "chr5": 180915260, 
        "chr6": 171115067, "chr7": 159138663, "chr8": 146364022, 
        "chr9": 141213431, "chrM": 16569, "chrMT": 16569,
        "chrX": 155270560, "chrY": 59373566}


if __name__ == "__main__":
    #args = get_args()
    pass



#import argparse, os, sys, warnings, time
#import numpy as np
#
#def get_args():
#    p = argparse.ArgumentParser()
#    return p.parse_args()
#
#if __name__ == "__main__":
#    args = get_args()
#



