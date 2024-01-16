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

import math
from collections import OrderedDict

import torch
from torch import Tensor, nn


class PartialSoftmax():
    @staticmethod
    def softmax(x, dim):
        """ Calculate softmax and return also the denuminator. """
        x_exp = torch.exp(x) # for numerical stability
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        return x_exp / x_exp_sum, torch.select(x_exp_sum, dim, 0)
        
    @staticmethod
    def partial_softmax(x, dim, den):
        """ Calculate partial softmax, in which `rest` is added to the denuminator. """
        x_exp = torch.exp(x) # for numerical stability
        return x_exp / torch.unsqueeze(den, dim)
    
    @staticmethod
    def naive_softmax(x, dim):
        """ Naive softmax implementation, as a sanity check. """
        x_exp = torch.exp(x) # for numerical stability
        x_exp_sum = torch.sum(x_exp, dim, keepdim=True)
        return x_exp / x_exp_sum
