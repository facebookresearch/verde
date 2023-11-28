# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
import argparse
import joblib
import numpy as np
import torch
import os
import logging

from src import utils
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp, create_logger
from src.generate.export import Generator
from src.generate.genSamples import BKZReducedRLWE, RA_Rb, BenchmarkBKZ
from multiprocessing import Process, Manager
from joblib import Parallel, delayed

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
    parser.add_argument("--resume_path", type=str, default="",
                        help="Path to load the checkpoints")
    parser.add_argument("--secret_dir", type=str, default="",
                        help="Path to the secret directory, for step Ab")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")
    parser.add_argument("--step", type=str, default="RA",
                        help="data generation step, RA, RA_tiny1, RA_tiny2, BKZ, or Ab")

    # iteration
    parser.add_argument("--env_base_seed", type=int, default=-1,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--batch_size", type=int, default=20,
                        help="Number of sentences per batch")
    parser.add_argument("--epoch_size", type=int, default=4000,
                        help="number of matrices to generate for each worker")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # Load data
    parser.add_argument("--reload_data", type=str, default="",
                        help="Directory to load the dataset from the disk. For RA_tiny, it's A; for Ab, it's R,A")
    parser.add_argument("--reload_perm", type=str, default="",
                        help="Directory to tinyA with permuted cols")
    parser.add_argument("--reload_size", type=int, default=100000,
                        help="Reloaded number of matrices")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", default=False, action="store_true", 
                        help="Enable all debug flags")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=True,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    parser.add_argument("--timeout", type=int, default=10,
                        help="Seconds before timeout in generate")
    parser.add_argument("--max_timeout_count", type=int, default=10,
                        help="Maximum Number of timeout in a row before stopping generation")
    
    # LWE
    parser.add_argument("--lwe", type=bool_flag, default=False, 
                        help="LWE if true. RLWE by default. Ignored by tiny samples, which loads from reload data")
    parser.add_argument("--N", type=int, default=-1, 
                        help="dimension of matrix")
    parser.add_argument("--Q", type=int, default=-1, 
                        help="modulo")
    parser.add_argument("--num_secret_seeds", type=int, default=5,
                        help="how many seeds for secrets")
    parser.add_argument("--min_hamming", type=int, default=3,
                        help="min hamming weight when generating secrets")
    parser.add_argument("--max_hamming", type=int, default=20,
                        help="max hamming weight when generating secrets. Used in step2_Ab and benchmarking")
    parser.add_argument("--secret_type", type=str, default="binary",
                        help="binary, ternary, gaussian, or binomial")
    parser.add_argument("--sigma", type=float, default=3, 
                        help='sigma for gaussian error')
    parser.add_argument("--gamma", type=int, default=2, 
                        help='gamma for binomial error')
    parser.add_argument("--correctQ", type=bool_flag, default=False, 
                        help='flip the Q range to be within -Q/2 and Q/2?')

    # LLL/BKZ reduction parameters
    parser.add_argument("--float_type", type=str, default="double", 
                        help="double, long double, dpe, dd, qd, or mpfr_<precision>")
    parser.add_argument("--lll_penalty", type=int, default=10, 
                        help="penalty on norm of LLL Reduced A")
    parser.add_argument("--algo", type=str, default='BKZ2.0', 
                        help='Phase 1, algorithm to use: BKZ or BKZ2.0')
    parser.add_argument("--lll_delta", type=float, default=0.96, 
                        help="Phase 1, hermite factor for LLL")
    parser.add_argument("--bkz_block_size", type=int, default=30, 
                        help="Phase 1, block size of the BKZ reduction")
    parser.add_argument("--algo2", type=str, default='BKZ2.0', 
                        help='Phase 2, algorithm to use: BKZ or BKZ2.0')
    parser.add_argument("--lll_delta2", type=float, default=0.99, 
                        help="Phase 2, hermite factor for LLL")
    parser.add_argument("--bkz_block_size2", type=int, default=40, 
                        help="Phase 2, block size of the BKZ reduction")
    parser.add_argument("--m", type=int, default=-1,
                        help="number of samples used in BKZ reduction, defaults to N")
    parser.add_argument("--threshold", type=float, default=-1,
                        help="the threshold to terminate reduction")
    parser.add_argument("--threshold2", type=float, default=0.5,
                        help="the threshold to go to phase 2")
    parser.add_argument("--rnorm", type=float, default=1.0, 
                        help='threshold for ||R||/q in step Ab')
    parser.add_argument("--use_polish", type=bool_flag, default=False, 
                        help='whether to polish after bkz')

    return parser
    
def initialize(params, logger, thread):
    if params.step in ["RA", "RA_tiny1", "RA_tiny2"]:
        sampleGen = BKZReducedRLWE(params, thread)
    elif params.step == "BKZ":
        if not os.path.isfile(os.path.join(params.dump_path, 'results.pkl')):
            keys = []
            for expNum in range(5): # fixed 5 experiments per hamming weight for now
                keys += [(expNum, params.max_hamming), (expNum, params.max_hamming - 1)]
            pickle.dump(dict([(key, []) for key in keys]), open(os.path.join(params.dump_path, 'results.pkl'), 'wb'))
        sampleGen = BenchmarkBKZ(params, thread)
    elif params.step == "Ab":
        sampleGen = RA_Rb(params)
    else:
        logger.info('Step not recognized. Must be one of RA, RA_tiny1, RA_tiny2, Ab, or BKZ')
        exit()

    generator = Generator(params, sampleGen)

    return generator

def get_data_one_worker(i, params):
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=0)
    # initialize generator
    gen = initialize(params, logger, i)
    # iteration
    gen.n_equations = 0
    
    while gen.n_equations < gen.epoch_size:
        if gen.export_data() is None:
            logger.info("============ End of file ============")
            gen.end_epoch()
            return
        gen.iter()
    
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
    utils.CUDA = not params.cpu

    if params.env_base_seed < 0: 
        params.env_base_seed = np.random.randint(1_000_000_000)

    n_cpu = joblib.cpu_count()
    n_jobs = min(n_cpu, params.num_workers)
    logger.info(f" Nb CPU: {n_cpu} and Nb worker: {params.num_workers}")
    Parallel(n_jobs=n_jobs, prefer='threads')(delayed(get_data_one_worker)(n, params) for n in range(n_jobs))

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
    
    if params.step in ["Ab", "BKZ"]:
        loaded_params = pickle.load(open(os.path.join(params.reload_data, 'params.pkl'), 'rb'))
        if type(loaded_params) != dict:
            loaded_params = loaded_params.__dict__
        params.N = loaded_params['N']
        params.Q = loaded_params['Q']
        if params.step == "BKZ":
            params.secret_type = loaded_params['secret_type'] if 'secret_type' in loaded_params else 'binary'
        else:
            if 'm' not in loaded_params or loaded_params['m'] == -1:
                params.m = loaded_params['N']
            else:
                params.m = loaded_params['m']
            params.tiny1 = loaded_params['step'] == 'RA_tiny1'
            params.tiny2 = loaded_params['step'] == 'RA_tiny2'
            params.orig_A_path = loaded_params['reload_data']

    # run experiment
    main(params)
