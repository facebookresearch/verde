# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import argparse
import numpy as np
import torch
import os
import pickle

import src
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp
from src.train.model import check_model_params, build_modules
from src.train.trainer import Trainer
from src.train.evaluator import Evaluator
from src.train.envs.generators import RLWE
from src.train.envs.lattice import LatticeEnvironment
from src.train.envs.datasets import create_dataloader

np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")
    
    # main parameters
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=True,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=2,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")
    
    # Hamming parameters
    parser.add_argument("--max_output_len", type=int, default=10, 
                        help="max length of output, beam max size")
    parser.add_argument("--hamming", type=int, default=4, 
                        help="Hamming weight of the secret")
    parser.add_argument("--secret_seed", type=int, default=0, 
                        help="Use dataset generated with this secret seed")

    # load data
    parser.add_argument("--reload_data", type=str, default="", 
                        help="The directory that has data.prefix to load dataset from the disk")
    parser.add_argument("--reload_size", type=int, default=10000000, 
                        help="Reloaded training set size, default large to load all data")
    parser.add_argument("--batch_load", type=bool_flag, default=False, 
                        help="Load training set by batches (of size reload_size)")
    parser.add_argument("--shuffle", type=bool_flag, default=True, 
                        help="Shuffle when loading the train data")
    parser.add_argument("--dim_red", type=str, default="", 
                        help="file to read columns for dimension reduction")
    parser.add_argument("--add_unred_perc", type=int, default=0, 
                        help="percentage of adding random linear combinations of original data to the train set")

    # Reuse samples
    parser.add_argument("--reuse", type=bool_flag, default=False, 
                        help='reuse samples during training?')
    parser.add_argument("--num_reuse_samples", type=int, default=10000, 
                        help='number of samples to choose from during one reuse batch')
    parser.add_argument("--times_reused", type=int, default=10, 
                        help='how many times to reuse a sample before discarding it?')

    # Bases
    parser.add_argument("--balanced_base", type=bool_flag, default=False, 
                        help="use balanced base?")
    parser.add_argument("--input_int_base", type=int, default=81, 
                        help="base of the input encoder")
    parser.add_argument("--output_int_base", type=int, default=0,
                        help="base of the output encoder")
    parser.add_argument("--correctQ", type=bool_flag, default=False, 
                        help='flip the Q range to be within -Q/2 and Q/2?')
    parser.add_argument("--share_token", type=int, default=1, 
                        help="if set to k, each k numbers at the less significant bit will share the same token")

    # model parameters
    parser.add_argument("--transformermode", type=str, default='old',
                        help="old for the old transformer")
    parser.add_argument("--enc_emb_dim", type=int, default=1024,
                        help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=512,
                        help="Decoder embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=1,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=2,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=4,
                        help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=4,
                        help="Number of Transformer decoder heads")
    parser.add_argument("--n_cross_heads", type=int, default=4,
                        help="Number of Transformer decoder heads in the cross attention")
    parser.add_argument("--n_enc_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer encoder")
    parser.add_argument("--n_dec_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer decoder")
    parser.add_argument("--xav_init", type=bool_flag, default=False,
                        help="Xavier initialization for transformer parameters")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="GELU initialization in FFN layers (else RELU)")
    
    parser.add_argument("--norm_attention", type=bool_flag, default=False,
                        help="Normalize attention and train temperature in Transformer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    
    # universal transformer parameters
    parser.add_argument("--enc_loop_idx", type=int, default=-1,
                        help="Index of the encoder shared weight layers (-1 for none)")
    parser.add_argument("--dec_loop_idx", type=int, default=0,
                        help="Index of the decoder shared weight layers (-1 for none)")
    parser.add_argument("--enc_loops", type=int, default=1,
                        help="Fixed/max nr of train passes through the encoder loop")
    parser.add_argument("--dec_loops", type=int, default=8,
                        help="Fixed/max nr of train passes through the decoder loop")
    parser.add_argument("--gated", type=bool_flag, default=True,
                        help="Gated loop layers")
    parser.add_argument("--enc_gated", type=bool_flag, default=False,
                        help="All encoder layers gated")
    parser.add_argument("--dec_gated", type=bool_flag, default=False,
                        help="All decoder layers gated")    
    parser.add_argument("--scalar_gate", type=bool_flag, default=False,
                        help="Scalar gates")

    # ACT
    parser.add_argument("--enc_act", type=bool_flag, default=False,
                        help="Encoder looped layer ACT")
    parser.add_argument("--dec_act", type=bool_flag, default=False,
                        help="Decoder looped layer ACT")
    parser.add_argument("--act_threshold", type=float, default=0.01,
                        help="Prob threshold for ACT")
    parser.add_argument("--act_ponder_coupling", type=float, default=0.05,
                        help="Ponder loss coupling for ACT")

    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of sentences per batch")
    parser.add_argument("--optimizer", type=str, default="adam_warmup,lr=0.00001,warmup_updates=8000,weight_decay=0.99",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--weighted_loss", type=bool_flag, default=False,
                        help='Weight loss to emphasize higher bits?')
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=2000000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=20,
                        help="Maximum number of epochs")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of CPU workers for DataLoader")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=True,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_eval_train", type=int, default=0,
                        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    parser.add_argument("--freeze_embeddings", type=bool_flag, default="False",
                        help="Freeze embeddings for retraining?")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_from_exp", type=str, default="",
                        help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="",
                        help="Path of data to eval")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")
    parser.add_argument("--eval_size", type=int, default=10000, 
                        help="Size of valid and test samples")
    parser.add_argument("--distinguisher_size", type=int, default=128, 
                        help="Size of distinguisher samples")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", default=False, help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--windows", type=bool_flag, default=False,
                        help="Windows version (no multiprocessing for eval)")
    parser.add_argument("--nvidia_apex", type=bool_flag, default=False,
                        help="NVIDIA version of apex")

    return parser


def parse_params(params):
    # load params of the dataset and add them to the params of the experiment
    paths = [os.path.join(params.reload_data, filename) for filename in ['train.prefix', 'test.prefix', 'params.pkl', 'secret.npy']]
    for path in paths: 
        assert os.path.isfile(path)
    params.train_path, params.test_path, params_path, secret_path = paths

    env_params = pickle.load(open(params_path, 'rb'))
    if type(env_params) != dict:
        env_params = env_params.__dict__
    params.N, params.Q, params.sigma = env_params['N'], env_params['Q'], env_params['sigma']
    params.secret_type = env_params['secret_type'] if 'secret_type' in env_params else 'binary'
    num_secret_seeds = env_params['num_secret_seeds'] if 'num_secret_seeds' in env_params else 1
    min_h = env_params['min_hamming'] if 'min_hamming' in env_params else 3
    secret = np.load(secret_path).T
    params.secret_col = (params.hamming-min_h) * num_secret_seeds + params.secret_seed  # secrets start at h=3
    params.secret = secret[params.secret_col]
    assert sum(params.secret != 0) == params.hamming
    dim_red_params(params)
    
    if params.env_base_seed < 0: 
        params.env_base_seed = np.random.randint(1_000_000_000)
    if params.output_int_base == 0:
        params.output_int_base = params.input_int_base

def dim_red_params(params):
    if params.dim_red == "":
        params.data_cols, params.dense_cols = None, None
        return
    assert os.path.isfile(params.dim_red)
    dim_red = pickle.load(open(params.dim_red, 'rb'))
    if type(dim_red) != dict:
        dim_red = dim_red.__dict__
    data_cols, dense_cols = dim_red[(params.reload_data, params.secret_seed, params.hamming)]
    params.data_cols, params.dense_cols = np.array(data_cols), np.array(dense_cols)
    # error if nonzeros are kicked out
    assert params.hamming == sum(params.secret[params.data_cols] != 0)
    # hamming weight reduction: flip the secret bits for the dense_cols
    if len(params.dense_cols) > 0:
        params.secret[params.dense_cols] -= 1
        params.secret[params.dense_cols] *= -1
    # dimension reduction: keep a subset of the columns
    params.secret = params.secret[params.data_cols]
    params.hamming = sum(params.secret != 0)
    params.N = len(params.data_cols)

def build_all(params):
    generator = RLWE(params)

    env = LatticeEnvironment(params, generator)
    modules = build_modules(env, params)

    if not params.eval_only:
        train_dataloader = create_dataloader(params, env, 'train')
    else:
        train_dataloader = None
    test_dataloader = create_dataloader(params, env, 'test')

    trainer = Trainer(params, modules, env, train_dataloader)
    evaluator = Evaluator(trainer, test_dataloader)

    return trainer, evaluator

def main(params):

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if params.cpu:
        assert not params.multi_gpu
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    trainer, evaluator = build_all(params)
    
    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # training
    while trainer.epoch < params.max_epoch:

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0
      
        while trainer.n_equations < trainer.epoch_size:
            trainer.enc_dec_step()
            trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        
        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.end_epoch(scores)
      



if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()
    parse_params(params)

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
