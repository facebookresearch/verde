# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import numpy as np
import torch

import src.train.envs.encoders as encoders


SPECIAL_WORDS = ["<eos>", "<pad>", "<mask>"]
logger = getLogger()

class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)

class LatticeEnvironment(object):
    def __init__(self, params, generator):
        self.generator = generator

        self.output_encoder = encoders.Encoder(params, True, output=True)
        self.input_encoder = encoders.Encoder(params, False, output=False)
        
        # vocabulary
        self.common_symbols = ['+']
        self.words = SPECIAL_WORDS + self.common_symbols + sorted(list(
            set(self.output_encoder.symbols + self.input_encoder.symbols )
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")


        # matrix counter for sample prefix
        self.mat_count = 0

        self.get_ids = np.vectorize(self._get_id)
        self.get_words = np.vectorize(self._get_word)    

    def input_to_infix(self, lst):
        m = self.input_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return str(m)

    def output_to_infix(self, lst):
        m = self.output_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return str(m)

    def _get_id(self, x):
        return self.word2id[x]
    
    def _get_word(self, id):
        return self.id2word[id]
    
    def decode_class(self, i):
        return "1"
   

    def code_class(self, xi, yi, int_len):
        return len(xi) // int_len

    def check_prediction(self, src, tgt, hyp):
        if len(hyp) == 0 or len(tgt) == 0:
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_hyp = self.output_encoder.decode(hyp)
        if val_hyp is None:
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_tgt = self.output_encoder.decode(tgt)
        if len(val_hyp) != len(val_tgt):
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_src = self.input_encoder.decode(src)
        return self.generator.evaluate(val_src, val_tgt, val_hyp), self.generator.evaluate_bitwise(tgt, hyp), self.generator.get_difference(val_tgt, val_hyp)
    

    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths
