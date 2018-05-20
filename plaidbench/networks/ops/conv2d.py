# Copyright 2018 Vertex.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import numpy as np      
import os
import random

from plaidbench.networks.ops import op

class Conv2d(op.Op):

    def __init__(self, params, ci, h, w, co, i, j):
        super(Conv2d, self).__init__(params)
        self.ci = ci
        self.h = h
        self.w = w
        self.co = co
        self.i = i
        self.j = j
        self.bs = params.batch_size

    def flops(self):
        flops = (self.i*self.j*self.co*2)*((self.h-self.i)*(self.w-self.j))*self.ci
        return flops

    def create_dataset_plaid(self):
        return np.zeros((self.params.epoch_size, self.h, self.w, self.ci), dtype='float')

    def build_model_plaid(self):
        # Plaid doesn't actually need the batch size to be bound yet
        from keras.layers import Conv2D, Input
        from keras.models import Model
        inp = Input(name='data', shape=(self.h, self.w, self.ci))
        op = Conv2D(self.co, (self.i, self.j), data_format='channels_last', use_bias=False)(inp)
        return Model(inp, op, name=self.get_key())                                               

    def get_key(self):
        return "conv2d" + "_".join(str(i) for i in ['conv2d', self.bs, self.co, self.h, self.w, self.ci, self.i, self.j])

    def get_tc_cache(self):
        path = os.path.join(self.params.backend_opts['tc_cachedir'], self.get_key())
        try:
            os.makedirs(path)
        except IOError:
            pass
        return path

    def create_dataset_tc(self):  
        # inp, weights
        import torch
        return torch.randn(self.bs, self.ci, self.h,self.w).cuda(), torch.randn(self.co, self.ci, self.i, self.j).cuda()

    def build_model_tc(self):
        import tensor_comprehensions as tc
        lang = """
        def convolution(float(N,CI,H,W) I, float(CO,CI,KH,KW) W1) -> (O) {
            O(n, co, h, w) +=! I(n, ci, h + kh, w + kw) * W1(co, ci, kh, kw)
        }
        """
        convolution = tc.define(lang, name="convolution")
        inp, kern = self.get_dataset()
        if (self.params.backend_opts['tc_autotune']):
            convolution.autotune(inp, kern, cache=self.get_tc_cache(), options=tc.Options("conv"),generations=self.params.backend_opts["tc_at_generations"], pop_size=self.params.backend_opts["tc_at_population"],elites=1,threads=8)
        return convolution

    def create_dataset_pyt(self):
        return self.create_dataset_tc()

    def build_model_pyt(self):
        import torch
        # Initialize the variables (i.e. assign their default value)
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True
        W = torch.randn(self.co, self.ci, self.i, self.j).cuda()
        conv = torch.nn.Conv2d(self.ci, self.co, (self.i, self.j), bias=False).cuda()
        conv.weights = W
        return lambda i, w: conv(i)

    def create_dataset_tvm(self):
        import tvm
        from topi.util import get_const_tuple
        I = tvm.placeholder((self.bs, self.ci, self.h, self.w), name='I')
        W = tvm.placeholder((self.co, self.ci, self.i, self.j), name='W')

        i_shape = get_const_tuple(I.shape)
        w_shape = get_const_tuple(W.shape)
        dtype = I.dtype
        i_np = np.zeros(shape=i_shape).astype(dtype)
        w_np = np.zeros(shape=w_shape).astype(dtype)
        return I, W, i_np, w_np
    
    def build_model_tvm(self):
        # Dummy build for timing
        import topi
        import tvm
        I, W, _, _ = self.get_dataset()
        device = self.params.backend_opts['tvm_driver']
        with tvm.target.create(device): 
            O = topi.nn.conv2d(I, W, 1, 1, layout='NCHW')
            tsched = topi.cuda.conv2d_nchw.schedule_conv2d_small_batch([O])
            with tvm.build_config(auto_unroll_max_step=1400, unroll_explicit=device != 'cuda'):
                return tvm.build(tsched, [I, W], device, name=self.get_key())

    def run_model_tvm(self, loops):
        # TODO: make this more reusable. TVM doesn't fit well with the abstractions as they are because of its use of contexts
        import topi
        import tvm
        from plaidbench import core
        device = self.params.backend_opts['tvm_driver']
        ctx = tvm.context(device, 0)
        with tvm.target.create(device): 
            I, W, i_np, w_np  = self.create_dataset_tvm()
            O = topi.nn.conv2d(I, W, 1, 1, layout='NCHW')
            tsched = topi.cuda.conv2d_nchw.schedule_conv2d_small_batch([O])
            i = tvm.nd.array(i_np, ctx)
            w = tvm.nd.array(w_np, ctx)
            with tvm.build_config(auto_unroll_max_step=1400, unroll_explicit=device != 'cuda'):
                op = tvm.build(tsched, [I, W], device, name=self.get_key())
            sw = core.StopWatch(False)
            sw.start()
            for _ in range(loops):
                op(i, w)
                ctx.sync()
            sw.stop()
            return sw.elapsed()