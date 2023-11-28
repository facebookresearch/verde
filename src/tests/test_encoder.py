# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import sys
sys.path.insert(0,'..')
from src.envs.encoders import Encoder


class ArgParseMock(object):
     def __init__(self, input_int_base = 81, output_int_base = 81, balanced_base = False, no_separator = True, Q = 251, N = 30):
         self.input_int_base = input_int_base
         self.output_int_base = output_int_base
         self.balanced_base = balanced_base
         self.no_separator = no_separator
         self.Q = Q
         self.N = N


class TestEncoderInputNotBalanced(unittest.TestCase):
    def setUp(self) -> None:
        params = ArgParseMock(input_int_base = 100, balanced_base = False)
        self.encoder = Encoder(params, single = False, output=False)
    
    def test_write_int_normal(self):
        input1 = 111
        self.assertEqual(self.encoder.write_int_normal(input1), ['1', '11']) 

    def test_encode_one_element(self):
        input1 = [100]
        self.assertEqual(self.encoder.encode(input1), ['1', '0'])

    def test_encode_multiple_elements(self):
        input1 = [1, 150]
        self.assertEqual(self.encoder.encode(input1), ['0', '1', '1', '50'])

    def test_parse_int(self):
        input1 = ['1', '0']
        self.assertEqual(self.encoder.parse_int(input1),(100, 2), "Should return the decoded integer and the end index")

    def test_decode(self,):
        input1 = ['0', '1']*self.encoder.dim
        self.assertEqual(self.encoder.decode(input1),[1]*self.encoder.dim)

        input2 = ['0', '1']
        self.assertIsNone(self.encoder.decode(input2), "Should return None because length of the list < 2dim")


class TestEncoderInputNotBalancedBase10(unittest.TestCase):
    def setUp(self) -> None:
        params = ArgParseMock(input_int_base = 10, balanced_base = False)
        self.encoder = Encoder(params, single = False, output=False)
    
    def test_write_int_normal(self):
        input1 = 111
        self.assertEqual(self.encoder.write_int_normal(input1), ['1', '1', '1']) 

    def test_encode_one_element(self):
        input1 = [100]
        self.assertEqual(self.encoder.encode(input1), ['1', '0', '0'])

    def test_encode_multiple_elements(self):
        input1 = [1, 150]
        self.assertEqual(self.encoder.encode(input1), ['0','0', '1', '1', '5', '0'])

    def test_parse_int(self):
        input1 = ['0', '1', '0']
        self.assertEqual(self.encoder.parse_int(input1),(10, 3), "Should return the decoded integer and the end index")

    def test_decode(self,):
        input1 = ['0', '0', '1']*self.encoder.dim
        self.assertEqual(self.encoder.decode(input1),[1]*self.encoder.dim)

        input2 = ['0', '1']
        self.assertIsNone(self.encoder.decode(input2), "Should return None because length of the list < 2dim")




class TestEncoderOutputNotBalanced(unittest.TestCase):
    def setUp(self) -> None:
        params = ArgParseMock(output_int_base = 100, balanced_base = False)
        self.encoder = Encoder(params, single = False, output=True)
        
    def test_encode_one_element(self):
        input1 = [100]
        self.assertEqual(self.encoder.encode(input1), ['1', '0'])

        
class TestEncoderInputBalanced(unittest.TestCase):
    def setUp(self) -> None:
        params = ArgParseMock(input_int_base = 100, balanced_base = True)
        self.encoder = Encoder(params, single = False, output=False)
        
    def test_write_int_balanced(self):
        input1 = 151
        self.assertEqual(self.encoder.write_int_balanced(input1), ['2','-49']) 

    def test_encode_one_element(self):
        input1 = [99]
        self.assertEqual(self.encoder.encode(input1), ['1', '-1'])

    def test_encode_multiple_elements(self):
        input2 = [1, 150]
        self.assertEqual(self.encoder.encode(input2), ['0', '1', '2', '-50'])
    
    def test_parse_int_balanced(self):
        input1 = ['1', '-1']
        self.assertEqual(self.encoder.parse_int(input1),(99, 2))
    




if __name__ == '__main__':
    unittest.main()
