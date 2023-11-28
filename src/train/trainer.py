# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import sys
import ast
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .optim import get_optimizer
from src.utils import to_cuda
from torch.utils.data import DataLoader

has_apex = True
try:
    import apex
except:
    has_apex = False


logger = getLogger()


class Trainer(object):
    def __init__(self, params, modules, env, train_dataloader):
        """
        Initialize trainer.
        """

        # modules / params
        self.modules = modules
        self.params = params
        self.env = env
        
        # epoch / iteration size
        self.epoch_size = params.epoch_size
        self.total_samples = 0
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        # assert not params.multi_gpu or params.amp == -1 or params.nvidia_apex
        assert not params.nvidia_apex or has_apex
        if params.multi_gpu: # and params.amp == -1:
            logger.info("Using torch.nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = torch.nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    find_unused_parameters=True,
                    broadcast_buffers=True,
                )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None
        if params.amp >= 0:
            self.init_amp()

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert len(split) == 2 and split[1].isdigit()
            self.decrease_counts_max = int(split[1])
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # Secret matching criterion -- stop if True
        self.secret_match = False

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m[1:], False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics
        }
        self.best_10percQ = 0

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0), ("processed_w", 0), ("loss", []), ("acc1", []), ("acc2", [])]
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # create data loaders
        if not params.eval_only:
            self.dataloader = train_dataloader
            self.iterator = iter(self.dataloader)


    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        mod_names = sorted(self.modules.keys())
        if params.nvidia_apex is True:
            modules, optimizer = apex.amp.initialize(
                [self.modules[k] for k in mod_names],
                self.optimizer,
                opt_level=("O%i" % params.amp),
            )
            self.modules = {k: module for k, module in zip(mod_names, modules)}
            self.optimizer = optimizer
        else:
            self.scaler = torch.cuda.amp.GradScaler()

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            exit()

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        # AMP optimization
        elif params.nvidia_apex is True:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(apex.amp.master_params(self.optimizer), params.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

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
        if self.n_total_iter % 50 != 0:
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

        # learning rates
        s_lr = (
            (" - LR: ")
            + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - MEM: {:.2f} GB - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
            torch.cuda.max_memory_allocated()/1024**3
        )
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            assert k in data
            try:
                v.load_state_dict(data[k])
            except:
                print('removing module prefix')
                if all([k2.startswith("module.") for k2 in data[k].keys()]):
                    data[k] = {
                        k2[len("module.") :]: v2 for k2, v2 in data[k].items()
                    }
                v.load_state_dict(data[k])

        
        # reload optimizer
        # AMP checkpoint reloading is buggy, we cannot reload optimizer
        # instead, we only reload current iterations / learning rates
        if self.params.amp == -1 or not self.params.nvidia_apex:
            logger.warning("Reloading checkpoint optimizer ...")
            self.optimizer.load_state_dict(data["optimizer"])
        else:
            logger.warning("Not reloading checkpoint optimizer.")
            for group_id, param_group in enumerate(
                self.optimizer.param_groups
            ):
                if "num_updates" not in param_group:
                    logger.warning("No 'num_updates' for optimizer.")
                    continue
                logger.warning(
                    "Reloading 'num_updates' and 'lr' for optimizer."
                )
                param_group["num_updates"] = data["optimizer"][
                    "param_groups"
                ][group_id]["num_updates"]
                param_group["lr"] = self.optimizer.get_lr_for_step(
                    param_group["num_updates"]
                )

        if self.params.fp16 and not self.params.nvidia_apex:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            print(self.scaler is None, "scaler" in data)
            assert (self.scaler is None) and ("scaler" not in data)

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
            
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                # self.save_checkpoint("best-%s" % metric)

        try:
        # Manually adding percs diff model saving. 
            currPercQ = float(ast.literal_eval(scores["valid_percs_diff"])[0])
            if currPercQ > self.best_10percQ:
                self.best_10percQ = currPercQ
                logger.info("New best score for tolerance=10percQ acc: %.6f" % (currPercQ))
                # self.save_checkpoint("best-10percQ")
        except:
            pass
        

    def end_epoch(self, scores):
        """
        End the epoch if secret found or stopping criterion is reached. 
        """
        # Check to see if secret_match is True -- this means the model has learned the secret based on eval tests.
        if self.secret_match:
            logger.info('Found secret match - ending experiment.')
            self.save_checkpoint("checkpoint")
            if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                os.system("scancel " + os.environ["SLURM_JOB_ID"])
            exit()


        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and self.params.is_master:
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric       

            factor = 1 if biggest else -1

            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()

        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self):
        """
        Return a training batch.
        """
        try:
            batch = next(self.iterator)
        except StopIteration:
            logger.info("Resetting data iterator")
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        except Exception as e:
            logger.error(
                "An unknown exception of type {0} occurred in line {1} when fetching batch. "
                "Arguments:{2!r}. Restarting ...".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args
                )
            )
            if self.params.is_slurm_job:
                if int(os.environ["SLURM_PROCID"]) == 0:
                    logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    def enc_dec_step(self):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules["encoder"], self.modules["decoder"]
        encoder.train()
        decoder.train()

        # batch
        (x1, len1), (x2, len2), _ = self.get_batch()

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = (
            alen[:, None] < len2[None] - 1
        )
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

        acc1 = -1
        acc2 = -1
        # forward / loss
        if params.amp == -1 or params.nvidia_apex:
            assert params.transformermode == 'old', "Other models not supported"
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
            )
            _, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False, weighted= self.params.weighted_loss,  # EJW -- set as param.
            )
        else:
            with torch.cuda.amp.autocast():
                if self.params.transformermode.startswith('new'):
                    loss, acc1, acc2, _ = encoder("fwd",x=x1,y=x2)
                else:
                    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                    decoded = decoder(
                        "fwd",
                        x=x2,
                        lengths=len2,
                        causal=True,
                        src_enc=encoded.transpose(0, 1),
                        src_len=len1,
                    )
                    word_scores, loss = decoder(
                        "predict",
                        tensor=decoded,
                        pred_mask=pred_mask,
                        y=y,
                        get_scores=False,
                        weighted= self.params.weighted_loss, 
                    )
                    with torch.no_grad():
                        t = torch.zeros_like(pred_mask, device=y.device)
                        t[pred_mask] += word_scores.max(1)[1] == y
                        valid1 = (t[0]).cpu().long()
                        valid2 = (t.sum(0) >= 2).cpu().long()
                        acc1 = valid1.float().mean().item()*100
                        acc2 = valid2.float().mean().item()*100

        self.stats['loss'].append(loss.item())
        self.stats['acc1'].append(acc1)
        self.stats['acc2'].append(acc2)

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats["processed_e"] += len1.size(0)
        self.stats["processed_w"] += (len1 + len2 - 2).sum().item()