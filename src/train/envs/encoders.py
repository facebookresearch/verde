# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
import numpy as np
import math


class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, params, single, output=False):
        self.int_base = params.input_int_base if not output else params.output_int_base
        self.share_token = params.share_token
        self.balanced = params.balanced_base
        self.new_encoding = params.new_encoding if hasattr(params, 'new_encoding') else False
        self.max_low_tokens =  math.ceil(self.int_base / self.share_token)
        self.int_len = math.floor(math.log(params.Q, self.int_base)) + 1
        self.Q = params.Q
        if self.balanced:
            max_digit = self.int_base // 2
            self.symbols = [str(i) for i in range(-max_digit, -max_digit + self.int_base)]
        else:
            self.symbols = [str(i) for i in range(self.max_low_tokens)]
            if self.new_encoding:
                self.symbols.extend([str(i) for i in range(self.max_low_tokens, self.max_low_tokens + math.ceil(self.Q/self.int_base))])
        
        self.dim = 1 if single else params.N

    def write_int(self, val):
        if self.balanced:
            return self.write_int_balanced(val)
        else:
            return self.write_int_normal(val)

    def write_int_normal(self, val):
        res = [(val // self.int_base**i) % self.int_base for i in range(self.int_len-1, -1, -1)]
        if self.new_encoding:
            res[0] += self.max_low_tokens
        return [str(res[0])] + [str(digit // self.share_token) for digit in res[1:]]

    def write_int_balanced(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -base//2 to (base-1)//2
        """
        base = self.int_base
        res = []
        max_digit = (base - 1) // 2
        for _ in range(self.int_len):
            rem = val % base
            val = val // base
            if rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
        return res[::-1]

    def parse_int(self, lst):
        res = 0
        for i in range(self.int_len):
            if i >= len(lst) or not (lst[i].isdigit() or lst[i][0] == '-' and lst[i][1:].isdigit()):
                return -self.Q, i
            if i==0 and self.new_encoding:
                res = res * self.int_base + int(lst[i]) - self.max_low_tokens
            else:
                res = res * self.int_base + int(lst[i])
        return res, self.int_len        

    def encode(self, x):
        if x.ndim ==2:
            return np.array([self._encode(row) for row in x])
        else:
            return np.array(self._encode(x))
            
    def _encode(self, vector):
        lst = []
        for val in vector:
            lst.extend(self.write_int(val))
        return lst

    def decode(self, lst):
        h = lst
        m = []
        for _ in range(self.dim):
            val, pos = self.parse_int(h)
            if val == -self.Q:
                return None
            h = h[pos:]
            m.append(val)
        return m
