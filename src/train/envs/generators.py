# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import circulant
from logging import getLogger

logger = getLogger()

class Generator(ABC):
    def __init__(self, params):
        self.N = params.N
        self.Q = params.Q
        self.secret = params.secret
        self.hamming = params.hamming
        self.sigma = params.sigma

    @abstractmethod
    def generate(self):
        pass

    @abstractmethod
    def evaluate(self, src, tgt, hyp):
        pass

class RLWE(Generator):
    def __init__(self, params):
        super().__init__(params)
        self.correctQ = params.correctQ
        self.q2_correction = np.vectorize(self.q2_correct)

    def generate(self, rng, train = True):
        # sample a uniformly from Z_q^n
        a = self.gen_a_row(rng, 0, self.Q)

        # do the circulant:
        c = self.compute_circulant(a)

        b = self.compute_b(c, rng)
       
        return c, b # return shapes NxN, N

    def gen_a_row(self, rng, minQ, maxQ):
        a = rng.randint(minQ, maxQ, size=self.N, dtype=np.int64)
        return a

    def compute_circulant(self, a):
        c = circulant(a)
        tri = np.triu_indices(self.N, 1)
        c[tri] *= -1
        if self.correctQ:
            c = self.q2_correction(c)

        c = c % self.Q

        assert (np.min(c) >= 0) and (np.max(c) < self.Q)

        return c
        
    def compute_b(self, c, rng):
        if self.sigma > 0:
            e = np.int64(rng.normal(0, self.sigma, size = self.N).round())
            b = (np.inner(c, self.secret) + e) % self.Q
        else:
            b = np.inner(c, self.secret) % self.Q

        if self.correctQ:
            b = self.q2_correction(b)
        return b

    def q2_correct(self, x):
        if x <= -self.Q/2:
            x = x+self.Q
        elif x >= self.Q/2:
            x = x-self.Q
        return x

    def evaluate(self, src, tgt, hyp):
        return 1 if hyp == tgt else 0

    def get_difference(self, tgt, hyp):
        diff = (hyp[0]-tgt[0]) % self.Q
        if diff > self.Q // 2:
            return abs(diff - self.Q)
        return diff

    def evaluate_bitwise(self, tgt, hyp):
        return [int(str(e1)==str(e2)) for e1,e2 in zip(tgt,hyp)]
