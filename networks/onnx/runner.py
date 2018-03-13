#!/usr/bin/env python

# Copyright 2017 Vertex.AI
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

import os.path

import onnx
if _backend_name == 'plaid':
    import onnx_plaidml.backend as backend
    GPU_DEVICE = 'GPU'
elif _backend_name == 'tf':
    import onnx_tf.backend as backend
    GPU_DEVICE = 'CUDA'
elif _backend_name == 'caffe2':
    import caffe2.python.onnx.backend as backend
    GPU_DEVICE = 'CUDA'


def scale_dataset(x_train):
    # No scaling needed when using standard ONNX test data
    return x_train


def build_model(full_path, onnx_cpu=False):
    model = onnx.load(full_path)
    if onnx_cpu:
        device = 'CPU'
    else:
        device = GPU_DEVICE
    return backend.prepare(model, device=device)
