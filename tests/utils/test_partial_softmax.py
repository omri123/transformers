# Copyright 2020 Omri Bloch. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest
import torch
from transformers.partial import PartialSoftmax


class TestPartialSoftmax(unittest.TestCase):
    def test_naive(self):
        x = torch.rand(100, 100)
        y1 = PartialSoftmax.naive_softmax(x, dim=1)
        y2 = torch.nn.functional.softmax(x, dim=1)
        self.assertTrue(torch.allclose(y1, y2))

    def test_naive_plus_denuminator(self):
        x = torch.rand(100, 100)
        y1, den1 = PartialSoftmax.softmax(x, dim=1)
        y2 = torch.nn.functional.softmax(x, dim=1)
        den2 = torch.exp(x).sum(dim=1)
        self.assertTrue(torch.allclose(y1, y2))
        self.assertTrue(torch.allclose(den1, den2))

    def test_partial_softmax_1d(self):
        x = torch.rand(30)
        _, den1 = PartialSoftmax.softmax(x[ 0:10], dim=0)
        _, den2 = PartialSoftmax.softmax(x[10:20], dim=0)
        _, den3 = PartialSoftmax.softmax(x[20:30], dim=0)
        den = den1 + den2 + den3
        y11 = PartialSoftmax.partial_softmax(x[ 0:10], dim=0, den=den)
        y12 = PartialSoftmax.partial_softmax(x[10:20], dim=0, den=den)
        y13 = PartialSoftmax.partial_softmax(x[20:30], dim=0, den=den)
        y1 = torch.cat((y11, y12, y13), dim=0)
        
        
        y2 = torch.nn.functional.softmax(x, dim=0)
        self.assertTrue(torch.allclose(y1, y2))
    
    def test_partial_softmax_2d(self):
        x = torch.rand(30, 10)
        _, den1 = PartialSoftmax.softmax(x[ 0:10, :], dim=0)
        _, den2 = PartialSoftmax.softmax(x[10:20, :], dim=0)
        _, den3 = PartialSoftmax.softmax(x[20:30, :], dim=0)
        den = den1 + den2 + den3
        y11 = PartialSoftmax.partial_softmax(x[ 0:10, :], dim=0, den=den)
        y12 = PartialSoftmax.partial_softmax(x[10:20, :], dim=0, den=den)
        y13 = PartialSoftmax.partial_softmax(x[20:30, :], dim=0, den=den)
        y1 = torch.cat((y11, y12, y13), dim=0)
        
        
        y2 = torch.nn.functional.softmax(x, dim=0)
        self.assertTrue(torch.allclose(y1, y2))
        
    def test_partial_softmax_2d_snd_dim(self):
        x = torch.rand(30, 10)
        _, den1 = PartialSoftmax.softmax(x[:,  0:10], dim=1)
        _, den2 = PartialSoftmax.softmax(x[:, 10:20], dim=1)
        _, den3 = PartialSoftmax.softmax(x[:, 20:30], dim=1)
        den = den1 + den2 + den3
        y11 = PartialSoftmax.partial_softmax(x[:,  0:10], dim=1, den=den)
        y12 = PartialSoftmax.partial_softmax(x[:, 10:20], dim=1, den=den)
        y13 = PartialSoftmax.partial_softmax(x[:, 20:30], dim=1, den=den)
        y1 = torch.cat((y11, y12, y13), dim=1)
        
        
        y2 = torch.nn.functional.softmax(x, dim=1)
        self.assertTrue(torch.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()
