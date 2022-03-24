
from typing import Dict, List, Union
from collections import OrderedDict
import numpy as np
from biock.logger import make_logger

logger = make_logger(title="")

class EarlyStopping(object):
    def __init__(self, \
            eval_keys: List[str], \
            score_keys: List[str], \
            loss_keys: List[str], \
            n_delay: int, \
            weight: Union[None, Dict[str, float]]=None, \
            patience: int=10
        ) -> None:
        """
        n_delay: supress early stop for the first n_delay epoches
        """
        if type(eval_keys) is str:
            eval_keys = [eval_keys]
        if type(score_keys) is str:
            score_keys = [score_keys]
        if type(loss_keys) is str:
            loss_keys = [loss_keys]
        assert len(set(eval_keys).difference(set(score_keys).union(set(loss_keys)))) == 0, "eval_key"

        self.__eval_keys = set(eval_keys)

        self.__losses = dict() # the less the better
        self.__scores = dict() # the bigger the better
        for k in score_keys:
            self.__scores[k] = OrderedDict()
        for k in loss_keys:
            self.__losses[k] = OrderedDict()

        if weight is not None:
            assert isinstance(weight, dict)
            assert self.__eval_keys.difference(set(weight.keys())) == 0
            self.__weight = weight
        else:
            self.__weight = {k: 1 for k in self.__eval_keys}

        if len(self.__eval_keys.intersection(set(self.__scores.keys()))) > 0:
            assert len(self.__eval_keys.intersection(set(self.__losses.keys()))) == 0
            self.__use_score = True
            self.__best_score = -np.Inf
            self.__best_loss = None
        else:
            assert len(self.__eval_keys.intersection(set(self.__losses.keys()))) > 0
            self.__use_score = False
            self.__best_loss = np.Inf
            self.__best_score = None

        self.__delay = n_delay
        self.patience = patience
        self.best_epoch = -1
        self.wait = 0
    
    def __str__(self):
        return "{}, scores: {}, losses: {}".format(
            type(self).__name__, self.__scores.keys(), self.__losses.keys())
    
    def update(self, epoch: int, **kwargs) -> bool:
        ## kwargs: {key: score/loss, xxx: xxx}
        ## return: continue ?
        values = list()
        for k in self.__scores:
            if k in kwargs:
                self.__scores[k][epoch] = kwargs[k]
            else:
                logger.warning("missing metric: {}".format(k))
                self.__scores[k][epoch] = None
        for k in self.__losses:
            if k in kwargs:
                self.__losses[k][epoch] = kwargs[k]
            else:
                logger.warning("missing metric: {}".format(k))
                self.__losses[k][epoch] = None

        for k, v in kwargs.items():
            if k in self.__eval_keys:
                values.append(self.__weight[k] * v)

        if (self.__use_score and np.mean(values) > self.__best_score):
            self.best_epoch = epoch
            self.__best_score = np.mean(values)
            self.wait = 0
            better = True
        elif not self.__use_score and np.mean(values) < self.__best_loss:
            self.best_epoch = epoch
            self.__best_loss = np.mean(values)
            self.wait = 0
            better = True
        else:
            if epoch >= self.__delay:
                self.wait += 1
            better = False
        return better
    
    def best_results(self, ):
        best_results = dict()
        best_results["epoch"] = self.best_epoch
        for k, v in self.__losses.items():
            best_results[k] = v[self.best_epoch]
        for k, v in self.__scores.items():
            best_results[k] = v[self.best_epoch]
        return best_results

