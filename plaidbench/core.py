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

from __future__ import division

from abc import ABCMeta, abstractmethod, abstractproperty
from collections import namedtuple
import enum
import errno
import json
import os
import time

import click
import numpy as np


class GoldenOutputNotAvailableError(Exception):
    pass


class ExtrasNeeded(click.ClickException):

    def __init__(self, extras):
        super(ExtrasNeeded, self).__init__(
            'Missing needed packages for benchmark; to fix, pip install {}'.format(
                ' '.join(extras)))
        self.extras = extras


class FrontendCommand(click.Command):

    def __init__(self, networks, *args, **kwargs):
        super(FrontendCommand, self).__init__(*args, **kwargs)
        self.__networks = networks

    def format_epilog(self, ctx, formatter):
        with formatter.section('Supported Networks'):
            formatter.write_text(', '.join(self.__networks))


class Precision(enum.Enum):
    TRAINING = 0.2
    INFERENCE = 5e-04


class StopWatch(object):

    def __init__(self, use_callgrind):
        self._start = None
        self._stop = None
        self._use_callgrind = use_callgrind
        self._callgrind_active = False
        self._total = 0.0

    def start_outer(self):
        # Like start(), but does not turn on callgrind.
        self._start = time.time()

    def start(self):
        self._start = time.time()
        if self._use_callgrind:
            os.system('callgrind_control --instr=on {}'.format(os.getpid()))
            self._callgrind_active = True

    def stop(self):
        if self._start is not None:
            stop = time.time()
            self._total += stop - self._start
            self._start = None
        if self._callgrind_active:
            self._callgrind_active = False
            os.system('callgrind_control --instr=off {}'.format(os.getpid()))

    def elapsed(self):
        return self._total


class Output(object):

    def __init__(self):
        self.contents = None
        self.precision = 'untested'


class Params(
        namedtuple('Params', ['batch_size', 'epochs', 'examples', 'warmups', 'network_name'])):
    """Parameters applied to a network during benchmarking."""
    __slots__ = ()

    @property
    def epoch_size(self):
        return self.examples // self.epochs


class ExplicitParamBuilder(object):
    """Builds Params for an explicit benchmark run."""

    def __init__(self, batch_size, epochs, examples, warmups=32):
        if not examples:
            examples = 1024
        self.params = Params(batch_size, epochs, examples, warmups, None)

    def __call__(self, frontend, network_names):
        if not network_names:
            raise click.UsageError('No networks specified; did you mean to add --blanket-run?')
        for network_name in network_names:
            params = self.params._replace(network_name=network_name)
            yield (frontend.model(params), params)


class BlanketParamBuilder(object):
    """Builds Params for a blanket benchmark run."""

    def __init__(self, epochs):
        self.params = Params(None, epochs, 256, 32, None)

    def __call__(self, frontend, network_names):
        if network_names:
            raise click.UsageError(
                'Networks specified with --blanket-run; choose one or the other')
        for network_name in frontend.network_names:
            for batch_size in frontend.blanket_batch_sizes:
                params = self.params._replace(network_name=network_name, batch_size=batch_size)
                yield (frontend.model(params), params)


class ConsoleReporter(object):

    def __init__(self):
        self.configuration = {}

    def report(self, params, results, output):
        print(results)

    def complete(self):
        pass


class ExplicitReporter(object):
    """Handles reports for an explicit benchmark run."""

    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.configuration = {}

    def report(self, params, results, output):
        try:
            os.makedirs(self.result_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                click.echo(ex)
                return
        with open(os.path.join(self.result_dir, 'result.json'), 'w') as out:
            json.dump(results, out)
        if isinstance(output, np.ndarray):
            np.save(os.path.join(self.result_dir, 'result.npy'), output)

    def complete(self):
        pass


class BlanketReporter(object):
    """Handles reports for a blanket benchmark run."""

    def __init__(self, result_dir):
        self.result_dir = result_dir
        self.outputs = {}
        self.configuration = {}

        self.configuration['plaid'] = None
        self.configuration['train'] = False
        self.configuration['blanket_run'] = True

    def report(self, params, results, output):
        composite_str = params.network_name + " : " + str(params.batch_size)
        self.outputs[composite_str] = dict(results)

    def complete(self):
        self.outputs['run_configuration'] = self.configuration
        try:
            os.makedirs(self.result_dir)
        except OSError as ex:
            if ex.errno != errno.EEXIST:
                click.echo(ex)
                return
        with open(os.path.join(self.result_dir, 'report.json'), 'w') as out:
            json.dump(self.outputs, out)


class Runner(object):
    """Runs an ML benchmark."""

    def __init__(self, param_builder=ExplicitParamBuilder(1, 2, 1024), reporter=ConsoleReporter()):
        """Initializes the benchmark runner.
        
        Args:
            param_builder ((frontend, [str])->((Model, Params)...)): A callable that takes a
                frontend and a list of network names, and returns a sequence of (Model, Params)
                tuples describing the benchmarks to be run.
            reporter (Reporter): Handles benchmark reports.
        """
        self.verbose = False
        self.result_dir = None
        self.callgrind = False
        self.param_builder = param_builder
        self.print_stacktraces = False
        self.reporter = reporter
        self.warmup = True

    def run(self, frontend, network_names):
        """Runs a set of benchmarks.
        
        Args:
            frontend (Frontend): The interface to the ML frontend.
            network_names ([str]): The names of the networks to benchmark.
        """
        # Stopwatch and Output initialization
        stop_watch = StopWatch(self.callgrind)
        compile_stop_watch = StopWatch(self.callgrind)
        exit_status = 0
        self.reporter.configuration['example_size'] = self.param_builder.params.examples

        for (model, params) in self.param_builder(frontend, network_names):
            click.echo('Running {0} examples with {1}, batch size {2}'.format(
                params.examples, params.network_name, params.batch_size))

            benchmark_results = {}
            model_output = None

            try:
                if params.examples % params.batch_size != 0:
                    raise ValueError('The number of examples must be divisible by the batch size.')

                model.validate()

                model.setup()

                click.echo('Compiling network...')
                compile_stop_watch.start_outer()
                stop_watch.start_outer()

                model.compile()
                model_output = model.run(once=True)

                compile_stop_watch.stop()

                # Run a few more warmups -- this seems to improve the variability of the
                # benchmark results.
                if self.warmup:
                    click.echo('Warming up ...')
                    model.run(warmup=True)

                click.echo('Main timing')
                stop_watch.start()
                model.run()
                stop_watch.stop()

                # Record stopwatch times
                execution_duration = stop_watch.elapsed()
                compile_duration = compile_stop_watch.elapsed()

                benchmark_results['compile_duration'] = compile_duration
                benchmark_results['execution_duration'] = execution_duration / params.examples
                benchmark_results['example_size'] = params.examples
                benchmark_results['batch_size'] = params.batch_size
                benchmark_results['model'] = params.network_name

                click.echo(
                    'Example finished, elapsed: {} (compile), {} (execution), {} (execution per example)'.
                    format(compile_duration, execution_duration,
                           execution_duration / params.examples))

                try:
                    (golden_output, precision) = model.golden_output()
                    (correct, max_error, max_abs_error, fail_ratio) = self._check_correctness(
                        golden_output, model_output, precision.value)
                    benchmark_results['correct'] = correct
                    benchmark_results['max_error'] = float(max_error)
                    benchmark_results['max_abs_error'] = float(max_abs_error)
                    benchmark_results['fail_ratio'] = fail_ratio
                    if correct:
                        status = 'PASS'
                    else:
                        status = 'FAIL'
                    click.echo(
                        'Correctness: {}, max_error: {}, max_abs_error: {}, fail_ratio: {}'.format(
                            status, max_error, max_abs_error, fail_ratio))
                except GoldenOutputNotAvailableError:
                    click.echo(
                        'Correctness: untested. Could not find golden data to compare against.')

            # Error handling
            except Exception as ex:
                # click.echo statements
                click.echo(ex)
                click.echo('Set --print-stacktraces to see the entire traceback')

                # Record error
                benchmark_results['exception'] = str(ex)

                # Set new exist status
                exit_status = -1

                # stacktrace loop
                if self.print_stacktraces:
                    raise

            self.reporter.report(params, benchmark_results, model_output)

        self.reporter.complete()
        return exit_status

    @staticmethod
    def _check_correctness(base_output, cur_output, precision):
        # TODO: Parameterize relative and absolute error tolerance
        correct = np.allclose(base_output, cur_output, rtol=precision, atol=1e-06)
        # This duplicates allclose calculation for more detailed report
        relative_error = ((precision * np.absolute(base_output - cur_output)) /
                          (1e-06 + precision * np.absolute(cur_output)))
        max_error = np.amax(relative_error)
        max_abs_error = np.amax(np.absolute(base_output - cur_output))
        correct_entries = 0
        incorrect_entries = 0
        for x in np.nditer(relative_error):
            if x > precision:
                incorrect_entries += 1
            else:
                correct_entries += 1
        try:
            fail_ratio = incorrect_entries / float(correct_entries + incorrect_entries)
        except ZeroDivisionError:
            fail_ratio = 'Undefined'

        return (correct, max_error, max_abs_error, fail_ratio)


class Frontend(object):
    """An abstract interface to an ML frontend."""
    __metaclass__ = ABCMeta

    def __init__(self, network_names):
        self.network_names = network_names

    @property
    def blanket_batch_sizes(self):
        return [1]

    @abstractmethod
    def model(self, params):
        """Returns a model built from the specified parameters.
        
        Args:
            params (Params): The parameters to use for the model.
        """
        pass


class Model(object):
    """An abstract interface to an ML model."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def setup(self):
        """Prepares a model to run benchmarks.

        This call gives an implementation to perform any setup/initialization actions that should
        not be included in the benchmark, e.g. downloading data files and other filesystem
        operations.
        """
        pass

    @abstractmethod
    def compile(self):
        """Compiles a model for repeated use.

        This call should be used by the implementation to construct an in-memory optimized form
        of the model, suitable for repeated use.  The implementation should not actually run
        the model; when compile() returns, the benchmarking infrastructure will issue an explicit
        call to run() to warm all relevant caches as part of the compilation measurement.
        """
        pass

    @abstractmethod
    def run(self, once, warmup):
        """Runs the model, e.g. performing an inference or training batch.

        Args:
            once (Boolean): If True, runs with a number of examples equal to the batch size. This
            is used in the very first run of a network for network compilation timing.

            warmup (Boolean): If True, uses the warmup parameter to determine the number of
            examples. This is used to prepare the graphics card and other variable-performance
            elements for the main timing run, ensuring that they dedicate the necessary resources
            to accurately time a heavy workload, without taking the time needed to run a full set
            of examples.

        Returns:
            The model outputs - if inference, the inference output; if training, the training loss.
        """
        pass

    def validate(self):
        """An optional hook for the model to use to validate its parameters."""
        pass

    @abstractmethod
    def golden_output(self):
        """The golden model output.
        
        Returns:
            (ndarray, Precision) - The golden model output.
            
        Throws:
            GoldenOutputNotAvailableError - If the golden output is unavailable for this model.
        """
