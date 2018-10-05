#!/usr/bin/env python
#
# Copyright 2018 Intel Corporation
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

from plaidbench import make_parser
import plaidbench.cli


def main():
    exit_status = 0
    parser = make_parser()
    args = parser.parse_args()

    argv = []

    # plaidbench arguments
    if args.verbose:
        argv.append('-{}'.format('v' * args.verbose))
    if args.result:
        argv.append('--result={}'.format(args.result))
    if args.callgrind:
        argv.append('--callgrind')
    if args.examples:
        argv.append('--examples={}'.format(args.examples))
    if args.epochs:
        argv.append('--epochs={}'.format(args.epochs))
    if args.batch_size:
        argv.append('--batch-size={}'.format(args.batch_size))
    if args.blanket_run:
        argv.append('--blanket-run')
    if args.no_warmup:
        argv.append('--no-warmup')
    if args.print_stacktraces:
        argv.append('--print-stacktraces')

    if args.onnx:
        # onnx arguments
        argv.append('onnx')
        if args.fp16:
            raise NotImplementedError(
                'With ONNX, --fp16 is defined by the model, not by the caller')
        if args.train:
            raise NotImplementedError('With ONNX, training vs. inference is model-specific')
        if args.onnx_cpu:
            argv.append('--cpu')
        if args.refresh_onnx_data:
            argv.append('--no-use-cached-data')
        if args.plaid or (not args.no_plaid and not args.caffe2):
            argv.append('--plaid')
        elif args.caffe2:
            argv.append('--caffe2')
        else:
            argv.append('--tensorflow')
    else:
        # keras arguments
        argv.append('keras')
        if args.fp16:
            argv.append('--fp16')
        if args.train:
            argv.append('--train')
        if args.onnx_cpu:
            raise NotImplementedError('--onnx_cpu is only meaningful with --onnx')
        if args.refresh_onnx_data:
            argv.append('--refresh-onnx-data is only meaningful with --onnx')
        if args.plaid or (not args.no_plaid and not args.caffe2):
            argv.append('--plaid')
        elif args.caffe2:
            raise ValueError('There is no Caffe2 backend for Keras')
        else:
            argv.append('--tensorflow')

    # Networks
    if args.module:
        argv.append(args.module)

    # Invoke plaidbench to do the actual benchmarking.
    plaidbench.cli.plaidbench(args=argv)


if __name__ == '__main__':
    main()
