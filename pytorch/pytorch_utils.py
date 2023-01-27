#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
import random
import typing
from typing import Literal, Optional, Tuple, List, Iterable, Dict, Any, Callable
import warnings
import logging
logger = logging.getLogger(__name__)

## genome one hot
# 'N': 0
# 'A': 1
# 'C': 2
# 'G': 3
# 'T': 4
ONEHOT = torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)

def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    else:
        return model

def filter_weights(model: torch.nn.Module, weights: typing.OrderedDict[str, torch.Tensor], ignore_keys: Iterable[str]=[]) -> typing.OrderedDict[str, torch.Tensor]:
    ignore_keys = set(ignore_keys)
    keys = list(weights.keys())
    model_keys = set(model.state_dict().keys())
    missing_keys = list()
    incompatible_keys = list()
    for k in keys:
        if k in ignore_keys:
            del weights[k]
        elif k not in model.state_dict():
            missing_keys.append(k)
            del weights[k]
        elif weights[k].size() != model.state_dict()[k].size():
            incompatible_keys.append((k, model.state_dict()[k].size(), weights[k].size()))
            del weights[k]
        else:
            model_keys.remove(k)
    if len(missing_keys) > 0:
        logger.warning("- missing keys (n={}): {}".format(len(missing_keys), missing_keys))
    if len(incompatible_keys):
        logger.warning("- incompatible keys(name, shape in model, shape in weights) (n={}): {}".format(len(incompatible_keys), incompatible_keys))
    return weights

def model_summary(model):
    """
    model: pytorch model
    """
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}


def make_onehot(N: int, include_zero: bool=False, dtype=torch.float) -> Tensor:
    onehot = torch.as_tensor(np.diag(np.ones(N)), dtype=dtype)
    if include_zero:
        onehot = torch.cat((torch.zeros((1, N)), onehot), dim=0),
    return onehot


def kl_divergence(mu1: Tensor, logvar1: Tensor, mu2: Optional[Tensor]=None, logvar2: Optional[Tensor]=None, reduction: Literal["mean"]="mean") -> Tensor:
    """
    $KL(q(z)||p(z)), q(z) \sim N(mu1, var1), p(z) \sim N(mu2, var2)$
    """
    if mu2 is None:
        mu2 = torch.zeros_like(mu1)
        logvar2 = torch.zeros_like(logvar1)
    kl_div = 0.5 * (logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1)
    kl_div = kl_div.mean(dim=1)
    if reduction == "mean":
        kl_div = torch.mean(kl_div)
    return kl_div


def set_seed(seed: int, amp: bool=True, force_deterministic: bool=False):
    if float(torch.version.cuda) >= 10.2:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if amp and not force_deterministic:
        logger.warning("torch.use_deterministic_algorithms is not True, program will run in nondeterministic mode")
    else:
        torch.use_deterministic_algorithms(True)



def get_device(model: nn.Module):
    return next(model.parameters()).device


def freeze_module(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module.eval()


def soft_cross_entropy(predicts, targets, reduction="mean"):
    assert reduction in {"mean", "none"}
    predicts_likelihood = F.log_softmax(predicts, dim=-1)
    targets_prob = F.softmax(targets, dim=-1)
    loss = -targets_prob * predicts_likelihood
    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss


def ce_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float=None,
    gamma: float=2,
    reduction: str = "none",
    **kwargs
) -> torch.Tensor:
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
    if alpha is not None:
        warnings.warn("ce_focal_loss does not support 'alpha'")
    shape = inputs.shape[:-1]
    ignore_index = kwargs.get("ignore_index", -100)
    dim = inputs.shape[-1]
    inputs = inputs.reshape(-1, dim)
    targets = targets.reshape(-1).long()
    keep = torch.where((targets >= 0) & (targets != ignore_index))[0]
    inputs = inputs[keep]
    targets = targets[keep]

    # tmp = torch.as_tensor(F.one_hot(targets.long(), num_classes=dim), dtype=inputs.dtype)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none", **kwargs)
    # p_t = p * F.one_hot(targets, num_classes=dim) #+ (1 - p) * (1 - F.one_hot(targets, num_classes=dim))
    p = torch.softmax(inputs, dim=-1) # (L, E)
    p_t = p[torch.arange(targets.shape[0]), targets] #* F.one_hot(targets, num_classes=dim) #+ (1 - p) * (1 - F.one_hot(targets, num_classes=dim))
    # p_t = p_t.sum(dim=-1)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # if alpha >= 0:
    #     alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    #     loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss.reshape(*shape)

    return loss


def get_lr_scheduler(name: str, config: Dict[str, Any]=None) -> Tuple[Callable, Dict, Dict]:
    if config is None:
        config = dict()
    scheduler_params = dict()
    control_opts = dict() # interval
    if name.lower() in {"cosine_lr", "cosineannealinglr"}:
        lr_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        scheduler_params["T_max"] = config.get("T_max", 64)
        scheduler_params["eta_min"] = config.get("eta_min", 0)
        scheduler_params["last_epoch"] = config.get("last_epoch", -1)
        control_opts["interval"] = config.get("interval", "step")
        control_opts["frequency"] = config.get("frequency", 1)
        if "monitor" in config:
            control_opts["monitor"] = config["monitor"]
    elif name.lower() in {"plateau", "reducelronplateau"}:
        lr_scheduler_class = torch.optim.lr_scheduler.ReduceLROnPlateau
        scheduler_params["mode"] = config["mode"]
        scheduler_params["factor"] = config["factor"]
        scheduler_params["patience"] = config["patience"]
        scheduler_params["threshold"] = config.get("threshold", 0.0001)
        scheduler_params["min_lr"] = config.get("min_lr", 1e-9)
        control_opts["interval"] = config.get("interval", "epoch")
        control_opts["frequency"] = config.get("frequency", 1)
    else:
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, name)
        if "interval" in config:
            control_opts["interval"] = config["interval"]
            del config["interval"]
        else:
            control_opts["interval"] = "epoch"
        if "frequency" in config:
            control_opts["frequency"] = config["frequency"]
            del config["frequency"]
        else:
            control_opts["frequency"] = 1
        scheduler_params = config
    return lr_scheduler_class, scheduler_params, control_opts
