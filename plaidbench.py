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

import argparse
import errno
import json
import numpy as np
import os
import sys
import time
import random

SUPPORTED_NETWORKS = ['inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception']


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


def has_plaid():
    try:
        import plaidml.keras
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
          raise ValueError('The number of examples per epoch must be greater than the batch size.')
      if examples % epochs != 0:
          raise ValueError('The number of examples must be divisible by the number of epochs.')
      if (examples // epochs) % batch_size != 0:
          raise ValueError('The number of examples per epoch is not divisible by the batch size.')


def train(x_train, y_train, epoch_size, model, batch_size, compile_stop_watch, epochs, stop_watch,
          output, network):
    # Training
    compile_stop_watch.start_outer()
    stop_watch.start_outer()

    run_initial(batch_size, compile_stop_watch, network, model)
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


def inference(network, model, batch_size, compile_stop_watch, output, x_train, examples,
              stop_watch):
    # Inference
    compile_stop_watch.start_outer()
    stop_watch.start_outer()

    run_initial(batch_size, compile_stop_watch, network, model)
    y = model.predict(x=x_train, batch_size=batch_size)

    compile_stop_watch.stop()

    output.contents = y

    for i in range(32 // batch_size + 1):
        y = model.predict(x=x_train, batch_size=batch_size)

    for i in range(examples // batch_size):
        stop_watch.start()
        y = model.predict(x=x_train, batch_size=batch_size)
        stop_watch.stop()

    stop_watch.stop()


def setup(train, epoch_size, batch_size):
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
        y_train_cats = None
        y_train = None
    return x_train, y_train


def load_model(module, x_train):
    this_dir = os.path.dirname(os.path.abspath(__file__))
    module = os.path.join(this_dir, 'networks', '%s.py' % module)
    globals = {}
    exec_(open(module).read(), globals)
    x_train = globals['scale_dataset'](x_train)
    model = globals['build_model']()
    printf("Model loaded.")
    return module, x_train, model


def load_golden(model, train, batch_size):
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


def run_initial(batch_size, compile_stop_watch, network, model):
    print("Compiling and running initial batch, batch_size={}".format(batch_size))
    optimizer = 'sgd'
    if network[:3] == 'vgg':
        from keras.optimizers import SGD
        optimizer = SGD(lr=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


def check_correctness(base_output, cur_output, precision):
    # TODO: Parameterize relative and absolute error tolerance
    if precision == 'high':
        rel_err = 1e-04
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


def main():
    exit_status = 0
    parser = argparse.ArgumentParser()
    plaidargs = parser.add_mutually_exclusive_group()
    plaidargs.add_argument('--plaid', action='store_true')
    plaidargs.add_argument('--no-plaid', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('--result', default='/tmp/plaidbench_results')
    parser.add_argument('--callgrind', action='store_true')
    parser.add_argument('-n', '--examples', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--blanket-run', action='store_true')
    parser.add_argument('--print-stacktraces', action='store_true')
    args1 = parser.parse_known_args()
    if args1[0].blanket_run == False:
        parser.add_argument('module', choices=SUPPORTED_NETWORKS)
    args = parser.parse_args()

    # Plaid, fp16, and verbosity setup
    if args.plaid or (not args.no_plaid and has_plaid()):
        printf('Using PlaidML backend.')
        import plaidml.keras
        if args.verbose:
            plaidml._internal_set_vlog(args.verbose)
        plaidml.keras.install_backend()
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

    batch_size = int(args.batch_size)
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
        networks = list(SUPPORTED_NETWORKS)
        batch_list = [1, 4, 8, 16]

        if args.plaid or (not args.no_plaid and has_plaid()):
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
                x, y_train = setup(args.train, epoch_size, batch_size)

                # Loading the model
                module, x, model = load_model(args.module, x)

                if args.train:
                    # training run
                    train(x, y_train, epoch_size, model, batch_size, compile_stop_watch,
                          epochs, stop_watch, output, network)
                else:
                    # inference run
                    inference(args.module, model, batch_size, compile_stop_watch, output, x,
                              examples, stop_watch)

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
                base_output = load_golden(args.module, args.train, batch_size)
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
