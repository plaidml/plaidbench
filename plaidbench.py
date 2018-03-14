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

from __future__ import print_function

from six import exec_
from six.moves.urllib.request import urlretrieve

import argparse
from collections import namedtuple
import errno
import hashlib
import json
import numpy as np
import os
import sys
import tarfile
import time
import random

SUPPORTED_NETWORKS = {
    'keras': ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception', 'imdb_lstm'],
    'onnx': [
        'bvlc_alexnet',
        'densenet121',
        'inception_v1',
        'inception_v2',
        'resnet50',
        'shufflenet',
        'squeezenet',  # TODO: Fix inputs/outputs (only available as *.pb)
        'vgg16',
        'vgg19',
    ],
}


def printf(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class StopWatch(object):

    def __init__(self, use_callgrind):
        self.__start = None
        self.__stop = None
        self.__use_callgrind = use_callgrind
        self.__callgrind_active = False
        self.__total = 0.0

    def start_outer(self):
        # Like start(), but does not turn on callgrind.
        self.__start = time.time()

    def start(self):
        self.__start = time.time()
        if self.__use_callgrind:
            os.system('callgrind_control --instr=on %d' % (os.getpid(),))
            self.__callgrind_active = True

    def stop(self):
        if self.__start is not None:
            stop = time.time()
            self.__total += stop - self.__start
            self.__start = None
        if self.__callgrind_active:
            self.__callgrind_active = False
            os.system('callgrind_control --instr=off %d' % (os.getpid(),))

    def elapsed(self):
        return self.__total


class Output(object):

    def __init__(self):
        self.contents = None
        self.precision = 'untested'


def has_plaid(frontend):
    if frontend == 'keras':
        try:
            import plaidml.keras
            return True
        except ImportError:
            return False
    if frontend == 'onnx':
        try:
            import onnx_plaidml
            return True
        except ImportError:
            return False


def value_check(train, examples, epochs, batch_size):
    if examples % batch_size != 0:
        raise ValueError('The number of examples must be divisible by the batch size.')
    if train:
        if epochs >= examples:
            raise ValueError('The number of epochs must be less than the number of examples.')
        if batch_size >= (examples // epochs):
            raise ValueError(
                'The number of examples per epoch must be greater than the batch size.')
        if examples % epochs != 0:
            raise ValueError('The number of examples must be divisible by the number of epochs.')
        if (examples // epochs) % batch_size != 0:
            raise ValueError(
                'The number of examples per epoch is not divisible by the batch size.')


def train(x_train, y_train, epoch_size, model, batch_size, compile_stop_watch, epochs, stop_watch,
          output, network):
    # Training
    compile_stop_watch.start_outer()
    stop_watch.start_outer()

    run_initial_keras(batch_size, compile_stop_watch, network, model)
    model.train_on_batch(x_train[0:batch_size], y_train[0:batch_size])

    compile_stop_watch.stop()

    x = x_train[:epoch_size]
    y = y_train[:epoch_size]

    for i in range(epochs):
        if i == 1:
            printf('Doing the main timing')
        stop_watch.start()
        history = model.fit(
            x=x, y=y, batch_size=batch_size, epochs=1, shuffle=False, initial_epoch=0)
        stop_watch.stop()
        time.sleep(.025 * random.random())
        if i == 0:
            output.contents = [history.history['loss']]
    output.contents = np.array(output.contents)
    stop_watch.stop()


def inference(frontend, network, model, batch_size, compile_stop_watch, output, x_train, examples,
              stop_watch):
    # Inference
    compile_stop_watch.start_outer()
    stop_watch.start_outer()

    y = run_initial(frontend, batch_size, network, model, x_train)

    compile_stop_watch.stop()

    output.contents = y

    for i in range(32 // batch_size + 1):
        predict_one(frontend, model, x_train, batch_size)

    for i in range(examples // batch_size):
        stop_watch.start()
        predict_one(frontend, model, x_train, batch_size)
        stop_watch.stop()

    stop_watch.stop()


def setup_cifar(train, epoch_size, batch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        printf('Loading the data')
        (x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
        x_train = x_train[:epoch_size]
        y_train_cats = y_train_cats[:epoch_size]
        y_train = to_categorical(y_train_cats, num_classes=1000)
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar_path = os.path.join(this_dir, 'cifar16.npy')
        x_train = np.load(cifar_path).repeat(1 + batch_size // 16, axis=0)[:batch_size]
        y_train = None
    return x_train, y_train


imdb_max_features = 20000
imdb_max_length = 80


def setup_imdb(train, epoch_size, batch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import imdb
        from keras.preprocessing import sequence
        printf('Loading the data')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=imdb_max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=imdb_max_length)
        x_train = x_train[:epoch_size]
        y_train = y_train[:epoch_size]
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        imdb_path = os.path.join(this_dir, 'imdb16.npy')
        x_train = np.load(imdb_path).repeat(1 + batch_size // 16, axis=0)[:batch_size]
        y_train = None
    return x_train, y_train


def _download_onnx_data(model, filename):
    # Uses the same directory structure as ONNX's backend test runner to reduce duplication
    expected_sha256 = {
        'bvlc_alexnet': '31202b69aa718b9d4aa67d0ed772efbe6886b69935cdb9224923c9ab8f36d01e',
        'densenet121': '6f3ec833eb27ef2170a407bc081c23893d2a1eb38f6374d36915e3ed1bba8242',
        'inception_v1': '3d934442c85cdeeb1cdceef83bd020dc19e3c8f6b3f5f97d5e7133aee0d41e40',
        'inception_v2': '1dba14b803bad006c591acbf8d5a9d459c139bc2067678a240fee4915d511bcf',
        'resnet50': '09076ac927e4a63730a02c3d40c9e1bb72fd88db942d203f713214a8b69cf09f',
        'shufflenet': 'a8d6339bf29c47d502cb8a11c3c753ec842f5b591f992b3af5590f06a437fd21',
        'squeezenet': 'c62a71fcb7b9944fd54ef4d6d19d065fe7989fe2101d28093b59bf19b5db7d7a',
        'vgg16': '52634b4dabb1255dfc0f48a2927dd04e9abf07b43e13d011457d9032d8088080',
        'vgg19': '4ec42e15829d47c47c1f2cf00fd91b116c1e2b47a3e6bd2323c9be72593d69ec',
    }
    onnx_home = os.path.expanduser(os.getenv('ONNX_HOME', os.path.join('~', '.onnx')))
    onnx_models = os.getenv('ONNX_MODELS', os.path.join(onnx_home, 'models'))
    model_dir = os.path.join(onnx_models, model)
    if not os.path.exists(os.path.join(model_dir, filename)):
        compressed_file = os.path.join(onnx_models, '{}.tar.gz'.format(model))
        url = 'https://s3.amazonaws.com/download.onnx/models/{}.tar.gz'.format(model)
        if not os.path.exists(compressed_file):
            if not os.path.exists(onnx_models):
                os.makedirs(onnx_models)
            print('Downloading {}...'.format(url), end='')
            sys.stdout.flush()
            try:
                urlretrieve(url, compressed_file)
            except:
                print("Failed to download data {} for {}".format(compressed_file, model))
                raise
            print('Done')
        print('Verifying checksum...', end='')
        sys.stdout.flush()
        with open(compressed_file, 'rb') as f:
            buffer_size = 65536
            hash = hashlib.sha256()
            while True:
                data = f.read(buffer_size)
                if not data:
                    break
                hash.update(data)
            if hash.hexdigest() != expected_sha256[model]:
                print("[WARNING] Invalid checksum on downloaded file from {}".format(url))
        print('Done')
        print('Extracting {}...'.format(compressed_file), end='')
        sys.stdout.flush()
        try:
            with tarfile.open(compressed_file) as f:
                f.extractall(onnx_models)
        except:
            print("Failed to extract data {} for {}".format(filename, model))
            raise
        print('Done')
    if not os.path.exists(os.path.join(model_dir, filename)):
        msg = ('Successfully retrieved model data but did not find the file {}. ' +
               'Check the filename or try clearing the cache at {}').format(filename, onnx_models)
        raise RuntimeError(msg)
    return os.path.join(model_dir, filename)


def setup_onnx_input(model):
    full_path = _download_onnx_data(model, 'test_data_0.npz')
    return np.load(full_path)['inputs'][0]


def load_model(module, frontend, backend, x_train, onnx_cpu=False):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    build_model_kwargs = dict()
    if frontend == 'onnx':
        filename = 'runner.py'
        full_path = _download_onnx_data(module, 'model.onnx')
        build_model_kwargs['full_path'] = full_path
        build_model_kwargs['onnx_cpu'] = onnx_cpu
    else:
        filename = '{}.py'.format(module)
    module_path = os.path.join(this_dir, 'networks', frontend, filename)
    globals = {'__file__': __file__, '_backend_name': backend}
    exec_(open(module_path).read(), globals)
    x_train = globals['scale_dataset'](x_train)
    model = globals['build_model'](**build_model_kwargs)
    printf("Model loaded.")
    return module_path, x_train, model


def load_golden(frontend, model, train, batch_size):
    if frontend == 'keras':
        if train:
            name = 'train'
        else:
            name = 'infer'
        filename = '{},bs-{}.npy'.format(name, batch_size)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        golden_path = os.path.join(this_dir, 'golden', model, filename)
        if not os.path.exists(golden_path):
            return None
        return np.load(golden_path)
    elif frontend == 'onnx':
        full_path = _download_onnx_data(model, 'test_data_0.npz')
        return np.load(full_path)['outputs'][0]
    else:
        raise ValueError('Unexpected frontend \'{}\''.format(frontend))


def run_initial(frontend, batch_size, network, model, x):
    print("Compiling and running initial batch, batch_size={}".format(batch_size))
    if frontend == 'keras':
        run_initial_keras(batch_size, network, model)
        y = model.predict(x=x, batch_size=batch_size)
    elif frontend == 'onnx':
        y = run_initial_onnx(batch_size, model, x)
    else:
        raise ValueError('Unrecognized frontend \'{}\''.format(frontend))
    return y


def run_initial_keras(batch_size, network, model):
    optimizer = 'sgd'
    if network[:3] == 'vgg':
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.0001)

    if network == 'imdb_lstm':
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def run_initial_onnx(batch_size, model, x):
    output = model.run([x[:batch_size]])
    return output


def predict_one(frontend, model, x, batch_size):
    if frontend == 'keras':
        y = model.predict(x=x, batch_size=batch_size)
    elif frontend == 'onnx':
        y = model.run([x[:batch_size]])
    else:
        raise ValueError('Unrecognized frontend \'{}\''.format(frontend))
    return y


def check_correctness(base_output, cur_output, precision):
    # TODO: Parameterize relative and absolute error tolerance
    if precision == 'high':
        rel_err = 5e-04
    elif precision == 'low':
        rel_err = 0.2

    correct = np.allclose(base_output, cur_output, rtol=rel_err, atol=1e-06)
    # This duplicates allclose calculation for more detailed report
    relative_error = ((rel_err * np.absolute(base_output - cur_output)) /
                      (1e-06 + rel_err * np.absolute(cur_output)))
    max_error = np.amax(relative_error)
    max_abs_error = np.amax(np.absolute(base_output - cur_output))
    correct_entries = 0
    incorrect_entries = 0
    for x in np.nditer(relative_error):
        if x > rel_err:
            incorrect_entries += 1
        else:
            correct_entries += 1
    try:
        fail_ratio = incorrect_entries / float(correct_entries + incorrect_entries)
    except ZeroDivisionError:
        fail_ratio = 'Undefined'

    return (correct, max_error, max_abs_error, fail_ratio)


def make_parser():
    # Create the parser outside of main() so the doc system can call this function
    # and thereby generate a web page describing these options. See docs/index.rst.
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true', help="Use PlaidML as the backend.")
    plaidargs.add_argument('--caffe2', action='store_true', help="Use Caffe2 as the backend.")
    plaidargs.add_argument(
        '--no-plaid',
        action='store_true',
        help="Use TensorFlow as the Keras backend instead of PlaidML.")
    frontendargs = parser.add_mutually_exclusive_group()
    frontendargs.add_argument('--keras', action='store_true', help='Use Keras as the frontend')
    frontendargs.add_argument('--onnx', action='store_true', help='Use ONNX as the frontend')
    parser.add_argument(
        '--fp16', action='store_true', help="Use half-precision floats, setting floatx='float16'.")
    parser.add_argument(
        '-v', '--verbose', action='count', default=0, help="Logging verbosity level (0..4).")
    parser.add_argument(
        '--result',
        default='/tmp/plaidbench_results',
        help="Destination directory for results output.")
    parser.add_argument(
        '--callgrind', action='store_true', help="Invoke callgrind during timing runs.")
    parser.add_argument(
        '-n', '--examples', type=int, default=None, help="Number of examples to use.")
    parser.add_argument('--epochs', type=int, default=2, help="Number of epochs per test.")
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument(
        '--train', action='store_true', help="Measure training performance instead of inference.")
    parser.add_argument(
        '--blanket-run',
        action='store_true',
        help="Run all networks at a range of batch sizes, ignoring the "
        "--batch-size and --examples options and the choice of network.")
    parser.add_argument(
        '--print-stacktraces',
        action='store_true',
        help="Print a stack trace if an exception occurs.")
    parser.add_argument(
        '--onnx-cpu', action='store_true', help='Use CPU instead of GPU (only used by ONNX)')
    all_supported_networks = set()
    for _, networks in SUPPORTED_NETWORKS.items():
        all_supported_networks = all_supported_networks.union(networks)
    parser.add_argument('module', choices=all_supported_networks, metavar='network')
    return parser


def main():
    exit_status = 0
    parser = make_parser()
    args = parser.parse_args()

    if args.onnx:
        frontend_name = 'onnx'
    else:
        # Keras is default
        frontend_name = 'keras'
    if args.module not in SUPPORTED_NETWORKS[frontend_name]:
        raise ValueError('The network {} is not supported with frontend {}'.format(
            args.module, frontend_name))

    # Plaid, fp16, and verbosity setup
    if args.onnx:
        # ONNX is handled specially for now
        if args.plaid or (not args.no_plaid and not args.caffe2 and has_plaid(frontend_name)):
            printf('Using PlaidML backend.')
            backend_name = 'plaid'
        elif args.caffe2:
            backend_name = 'caffe2'
        else:
            backend_name = 'tf'
        if args.fp16:
            raise NotImplementedError("FP16 not yet integrated with ONNX frontend")
        if args.verbose:
            raise NotImplementedError("Verbosity not yet integrated with ONNX frontend")
        if not args.blanket_run and args.batch_size != 1:
            raise NotImplementedError("Only batch size 1 currently supported with ONNX frontend")
        if args.train:
            raise NotImplementedError("Can only test inference with ONNX frontend")
    else:
        # Default is Keras
        if args.plaid or (not args.no_plaid and has_plaid(frontend_name)):
            printf('Using PlaidML backend.')
            import plaidml.keras
            if args.verbose:
                plaidml._internal_set_vlog(args.verbose)
            plaidml.keras.install_backend()
            backend_name = 'plaid'
        else:
            backend_name = 'tf'
        if args.fp16:
            from keras.backend import set_floatx
            set_floatx('float16')

    if args.examples is None:
        if args.blanket_run:
            examples = 256
        else:
            examples = 1024
    else:
        examples = args.examples

    epochs = args.epochs
    epoch_size = examples // epochs
    networks = []
    batch_list = []
    output = Output()

    # Stopwatch and Output initialization
    stop_watch = StopWatch(args.callgrind)
    compile_stop_watch = StopWatch(args.callgrind)

    # Blanket run - runs every supported network
    if args.blanket_run:
        data = {}
        outputs = {}
        networks = list(SUPPORTED_NETWORKS[frontend_name])
        if frontend_name == 'onnx':
            batch_list = [1]
        else:
            batch_list = [1, 4, 8, 16]

        if args.plaid or (not args.no_plaid and has_plaid(frontend_name)):
            import plaidml
            data['plaid'] = plaidml.__version__
        else:
            data['plaid'] = None

        data['example_size'] = examples
        data['train'] = args.train
        data['blanket_run'] = True
        outputs['run_configuration'] = data.copy()
    else:
        networks.append(args.module)
        batch_list.append(args.batch_size)

    for network in networks:
        printf()
        printf("Current network being run : " + network)
        args.module = network
        network_data = {}

        for batch in batch_list:
            batch_size = batch
            printf('Running {0} examples with {1}, batch size {2}'.format(
                examples, network, batch))
            value_check(args.train, examples, epochs, batch_size)

            # Run network w/ batch_size
            try:
                # Setup
                if frontend_name == 'onnx':
                    # No y_train b/c PlaidBench does not support ONNX training
                    x = setup_onnx_input(args.module)
                elif network == 'imdb_lstm':
                    x, y_train = setup_imdb(args.train, epoch_size, batch_size)
                else:
                    x, y_train = setup_cifar(args.train, epoch_size, batch_size)

                # Loading the model
                module, x, model = load_model(args.module, frontend_name, backend_name, x,
                                              args.onnx_cpu)

                if args.train:
                    # training run
                    train(x, y_train, epoch_size, model, batch_size, compile_stop_watch, epochs,
                          stop_watch, output, network)
                else:
                    # inference run
                    inference(frontend_name, args.module, model, batch_size, compile_stop_watch,
                              output, x, examples, stop_watch)

                # Record stopwatch times
                execution_duration = stop_watch.elapsed()
                compile_duration = compile_stop_watch.elapsed()

                network_data['compile_duration'] = compile_duration
                network_data['execution_duration'] = execution_duration / examples
                network_data['precision'] = output.precision
                network_data['example_size'] = examples
                network_data['batch_size'] = batch_size
                network_data['model'] = network

                # Print statement
                printf(
                    'Example finished, elapsed: {} (compile), {} (execution), {} (execution per example)'.
                    format(compile_duration, execution_duration, execution_duration / examples))

                # Attempt to validate correctness
                base_output = load_golden(frontend_name, args.module, args.train, batch_size)
                if base_output is not None:
                    if args.train:
                        precision = 'low'
                    else:
                        precision = 'high'
                    (correct, max_error, max_abs_error, fail_ratio) = check_correctness(
                        base_output, output.contents, precision)
                    network_data['correct'] = correct
                    network_data['max_error'] = float(max_error)
                    network_data['max_abs_error'] = float(max_abs_error)
                    network_data['fail_ratio'] = fail_ratio
                    if correct:
                        status = 'PASS'
                    else:
                        status = 'FAIL'
                    printf(
                        'Correctness: {}, max_error: {}, max_abs_error: {}, fail_ratio: {}'.format(
                            status, max_error, max_abs_error, fail_ratio))
                else:
                    printf('Correctness: untested. Could not find golden file to compare against.')

            # Error handling
            except Exception as ex:
                # Print statements
                printf(ex)
                printf('Set --print-stacktraces to see the entire traceback')

                # Record error
                network_data['exception'] = str(ex)

                # Set new exist status
                exit_status = -1

                # stacktrace loop
                if args.print_stacktraces:
                    raise

            # stores network data in dictionary
            if args.blanket_run:
                composite_str = network + " : " + str(batch_size)
                outputs[composite_str] = dict(network_data)
            # write all data to result.json / report.npy if single run
            else:
                try:
                    os.makedirs(args.result)
                except OSError as ex:
                    if ex.errno != errno.EEXIST:
                        printf(ex)
                        return
                with open(os.path.join(args.result, 'result.json'), 'w') as out:
                    json.dump(network_data, out)
                if isinstance(output.contents, np.ndarray):
                    np.save(os.path.join(args.result, 'result.npy'), output.contents)
                # close
                sys.exit(exit_status)

    # write all data to report.json if blanket run
    if args.blanket_run:
        try:
            os.makedirs(args.result)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                printf(ex)
                return
        with open(os.path.join(args.result, 'report.json'), 'w') as out:
            json.dump(outputs, out)
    # close
    sys.exit(exit_status)


if __name__ == '__main__':
    main()
