# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import unittest
import sys
import numpy as np
sys.path.insert(0,'..')

from src.generators import RLWE

class RLWEArgParseMock(object):
    def __init__(self,  Q = 251, N = 10, hamming = 5):

        self.secret = ""
        self.Q = Q
        self.N = N
        self.hamming = hamming
        self.sparsity = 0.5
        self.sigma = 3
        self.correctQ = False
        self.reuse = True
        self.num_reuse_samples = 100
        self.times_reused = 10
        self.K = 1
        self.secrettype = "b"
        self.env_base_seed = 0



class TestRLWE(unittest.TestCase):
    def setUp(self) -> None:
        self.params = RLWEArgParseMock()
        self.rng = np.random.default_rng(self.params.env_base_seed)
        self.generator = RLWE(self.params, self.rng)
    
    def test_genSecretKey_binary_hamming_positive(self):
        N = self.params.N
        secrettype = "b"

        secret = self.generator._gen_secret_key(secrettype)
        self.assertEqual(len(secret),N)
        self.assertEqual(set(secret), {0,1})
    
    def test_genSecretKey_other_types_hamming_positive(self):
        N = self.params.N
        secrettype = "g"
        secret = self.generator._gen_secret_key(secrettype)
        self.assertEqual(len(secret),N)

        secrettype = "u"
        secret = self.generator._gen_secret_key(secrettype)
        self.assertEqual(len(secret),N)

        secrettype = "t"
        secret = self.generator._gen_secret_key(secrettype)
        self.assertEqual(len(secret),N)

        secrettype = ""
        with self.assertRaises(UnboundLocalError):
            self.generator._gen_secret_key(secrettype)
    

if __name__ == '__main__':
    unittest.main()


        
