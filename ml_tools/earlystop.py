
from typing import Any, Dict, List, Union
from collections import OrderedDict
import numpy as np
from biock.logger import make_logger
import copy

logger = make_logger(title="")

class EarlyStopping(object):
    def __init__(self, \
            eval_keys: List[str], \
            n_delay: int, \
            score_keys: List[str]=list(), \
            loss_keys: List[str]=list(), \
            weight: Union[None, Dict[str, float]]=None, \
            patience: int=10,
            threshold=1E-4
        ) -> None:
        """
        n_delay: supress early stop for the first n_delay epoches
        """
        assert len(score_keys) + len(loss_keys) > 0, "both score_keys and loss_keys are None"
        if type(eval_keys) is str:
            eval_keys = [eval_keys]
        if type(score_keys) is str:
            score_keys = [score_keys]
        if type(loss_keys) is str:
            loss_keys = [loss_keys]
        assert len(set(eval_keys).difference(set(score_keys).union(set(loss_keys)))) == 0, "eval_key"

        self.__eval_keys = set(eval_keys)
        self.threshold = threshold

        self.__losses = dict() # the less the improved
        self.__scores = dict() # the bigger the improved
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
        self.go_on = True
        self.improved = True
    
    def state_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)
    
    def load_state_dict(self, d: Dict[str, Any]):
        for k, v in d.items():
            self.__dict__[k] = v

    
    def __str__(self):
        return "{}, scores: {}, losses: {}".format(
            type(self).__name__, self.__scores.keys(), self.__losses.keys())
    
    def update(self, epoch: int, **kwargs):
        ## kwargs: {key: score/loss, xxx: xxx}
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

        if self.__use_score and np.mean(values) > self.__best_score + self.threshold: # improved score, go on
            self.best_epoch = epoch
            self.__best_score = np.mean(values)
            self.wait = 0
            self.improved = True
            self.go_on = True
        elif not self.__use_score and np.mean(values) < self.__best_loss - self.threshold: # decreased loss, go on
            self.best_epoch = epoch
            self.__best_loss = np.mean(values)
            self.wait = 0
            self.improved = True
            self.go_on = True
        else:
            self.improved = False
            self.wait += 1
            if epoch < self.__delay:
                self.wait = 0
                self.go_on = True
            elif self.wait < self.patience:
                self.go_on = True
            else:
                self.go_on = False
    
    @property
    def best_results(self):
        best_results = dict()
        best_results["epoch"] = self.best_epoch
        for k, v in self.__losses.items():
            best_results[k] = v[self.best_epoch]
        for k, v in self.__scores.items():
            best_results[k] = v[self.best_epoch]
        return best_results

