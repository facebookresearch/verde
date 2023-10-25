import os
import io
import sys
import ast
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn

from ..utils import to_cuda, timeout, TimeoutError


logger = getLogger()


class Generator(object):
    def __init__(self, params, gen):
        """
        Initialize trainer.
        """
        # params
        self.params = params
        self.gen = gen
        self.print_cycle = 500 if params.step == "Ab" else 1
        
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        assert self.epoch_size > 0

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.timeout_count = 0
        self.total_count = 0
        self.stats = { "processed_e": 0 }
        self.last_time = time.time()

        # file handler to export data
        export_path_prefix = os.path.join(params.dump_path, "data.prefix")
        if params.step == 'Ab':
            export_path_prefix = os.path.join(params.dump_path, "test.prefix")
        self.file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")
        logger.info(f"Data will be stored in: {export_path_prefix} ...")

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % self.print_cycle != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
        )
        self.stats["processed_e"] = 0
        self.last_time = new_time

        # log speed + stats 
        logger.info(s_iter + s_speed + s_stat)

    def end_epoch(self):
        """
        End the epoch. 
        """
        if self.params.step == 'Ab':
            np.save(os.path.join(self.params.dump_path, 'diff.npy'), self.gen.diff)
            logger.info(f'Saved diff at {os.path.join(self.params.dump_path, "diff.npy")}')

    def export_data(self):
        """
        Export data to the disk.
        """
        while True:
            try:
                sample = self.gen.generate()
                break
            except TimeoutError:
                logger.info(f'Timeout, count: {self.timeout_count}')
                self.timeout_count += 1
                if self.timeout_count >= self.params.max_timeout_count:
                    sample = None
                    break
            except Exception as e:
                logger.info(f'An exception happened in the generator: {e}')
                continue
        self.timeout_count = 0

        if sample is None:
            return None
        else:
            X, Y = sample
            assert X.shape[0] == Y.shape[0]
            for i in range(X.shape[0]):
                prefix1_str = " ".join(X[i].astype(str))
                prefix2_str = " ".join(Y[i].astype(str))
                self.file_handler_prefix.write(f"{prefix1_str} ; {prefix2_str}\n")
                self.total_count += 1

                if self.total_count == 10000 and self.params.step == 'Ab':
                    self.file_handler_prefix.flush()
                    export_path_prefix = os.path.join(self.params.dump_path, "train.prefix")
                    self.file_handler_prefix = io.open(export_path_prefix, mode="a", encoding="utf-8")

            self.file_handler_prefix.flush()

        # number of processed sequences / words
        self.n_equations += 1
        self.stats["processed_e"] += 2*self.params.N
        return True
