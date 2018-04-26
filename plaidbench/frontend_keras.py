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

import importlib
import os

import click
import numpy as np

import plaidml

from plaidbench import core


def setup_cifar(train, epoch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import cifar10
        from keras.utils.np_utils import to_categorical
        click.echo('Loading CIFAR data')
        (x_train, y_train_cats), (_, _) = cifar10.load_data()
        x_train = x_train[:epoch_size]
        y_train_cats = y_train_cats[:epoch_size]
        y_train = to_categorical(y_train_cats, num_classes=1000)
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cifar_path = os.path.join(this_dir, 'cifar16.npy')
        x_train = np.load(cifar_path).repeat(1 + epoch_size // 16, axis=0)[:epoch_size]
        y_train = None
    return x_train, y_train


imdb_max_features = 20000
imdb_max_length = 80


def setup_imdb(train, epoch_size):
    # Setup
    if train:
        # Training setup
        from keras.datasets import imdb
        from keras.preprocessing import sequence
        click.echo('Loading IMDB data')
        (x_train, y_train), (_, _) = imdb.load_data(num_words=imdb_max_features)
        x_train = sequence.pad_sequences(x_train, maxlen=imdb_max_length)
        x_train = x_train[:epoch_size]
        y_train = y_train[:epoch_size]
    else:
        # Inference setup
        this_dir = os.path.dirname(os.path.abspath(__file__))
        imdb_path = os.path.join(this_dir, 'imdb16.npy')
        x_train = np.load(imdb_path).repeat(1 + epoch_size // 16, axis=0)[:epoch_size]
        y_train = None
    return x_train, y_train


class Model(core.Model):

    def __init__(self, frontend, params):
        self.frontend = frontend
        self.params = params

    def setup(self):
        if self.params.network_name == 'imdb_lstm':
            x, y_train = setup_imdb(self.frontend.train, self.params.epoch_size)
        else:
            x, y_train = setup_cifar(self.frontend.train, self.params.epoch_size)
        self.x = x
        self.y_train = y_train

        this_dir = os.path.dirname(os.path.abspath(__file__))
        build_model_kwargs = dict()
        filename = '{}.py'.format(self.params.network_name)
        module_path = os.path.join(this_dir, 'networks', 'keras', filename)
        mod = {'__file__': __file__, '_backend_name': self.frontend.backend_name}
        with open(module_path) as f:
            code = compile(f.read(), module_path, 'exec')
            eval(code, mod)
        self.x = mod['scale_dataset'](self.x)
        self.model = mod['build_model'](**build_model_kwargs)
        click.echo('Model loaded.')

    def compile(self):
        if self.params.network_name[:3] == 'vgg':
            from keras.optimizers import SGD
            optimizer = SGD(lr=0.0001)
        else:
            optimizer = 'sgd'

        if self.params.network_name == 'imdb_lstm':
            self.model.compile(
                optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(
                optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    def keras_golden_output(self, typename):
        filename = '{},bs-{}.npy'.format(typename, self.params.batch_size)
        this_dir = os.path.dirname(os.path.abspath(__file__))
        golden_path = os.path.join(this_dir, 'golden', self.params.network_name, filename)
        if not os.path.exists(golden_path):
            raise core.GoldenOutputNotAvailableError()
        return np.load(golden_path)


class InferenceModel(Model):

    def run(self, once=False, warmup=False):
        if once:
            epoch_size = self.params.batch_size
        elif warmup:
            epoch_size = self.params.warmups
        else:
            epoch_size = self.params.epoch_size
        return self.model.predict(x=self.x[:epoch_size], batch_size=self.params.batch_size)

    def golden_output(self):
        return (self.keras_golden_output('infer'), core.Precision.INFERENCE)


class TrainingModel(Model):

    def validate(self):
        if self.params.examples % self.params.epochs != 0:
            raise ValueError('The number of examples must be divisible by the number of epochs.')
        if self.params.examples < self.params.epochs:
            raise ValueError(
                'The number of examples must be greater than or equal to the number of epochs (examples-per-epoch must be >= 1).'
            )
        if (self.params.examples // self.params.epochs) < self.params.batch_size:
            raise ValueError(
                'The number of examples per epoch must be greater than or equal to the batch size.'
            )
        if (self.params.examples // self.params.epochs) % self.params.batch_size != 0:
            raise ValueError(
                'The number of examples per epoch is not divisible by the batch size.')

    def run(self, once=False, warmup=False):
        if once:
            epoch_size = self.params.batch_size
            epochs = 1
        elif warmup:
            epoch_size = self.params.warmups
            epochs = 1
        else:
            epoch_size = self.params.epoch_size
            epochs = self.params.epochs
        history = self.model.fit(
            x=self.x[:epoch_size],
            y=self.y_train[:epoch_size],
            batch_size=self.params.batch_size,
            epochs=epochs,
            shuffle=False,
            initial_epoch=0)
        return np.array(history.history['loss'])

    def golden_output(self):
        return (self.keras_golden_output('train'), core.Precision.TRAINING)


class Frontend(core.Frontend):
    NETWORK_NAMES = [
        'inception_v3', 'mobilenet', 'resnet50', 'vgg16', 'vgg19', 'xception', 'imdb_lstm'
    ]

    def __init__(self, backend_name, fp16, train):
        super(Frontend, self).__init__(Frontend.NETWORK_NAMES)
        self.backend_name = backend_name
        self.fp16 = fp16
        self.train = train

    def model(self, params):
        if self.train:
            return TrainingModel(self, params)
        return InferenceModel(self, params)

    @property
    def blanket_batch_sizes(self):
        return [1, 4, 8, 16]


@click.command(cls=core.FrontendCommand, networks=Frontend.NETWORK_NAMES)
@click.option(
    '--plaid', 'backend', flag_value='plaid', default=True, help='Use PlaidML as the backend')
@click.option(
    '--tensorflow', 'backend', flag_value='tensorflow', help='Use TensorFlow as the backend')
@click.option(
    '--fp16/--no-fp16',
    default=False,
    help='Use half-precision floats, settings floatx=\'float16\'')
@click.option(
    '--train/--no-train', default=False, help='Measure training performance instead of inference')
@click.argument('networks', nargs=-1, type=click.Choice(Frontend.NETWORK_NAMES))
@click.pass_context
def cli(ctx, backend, fp16, train, networks):
    """Benchmarks Keras neural networks."""
    runner = ctx.ensure_object(core.Runner)
    frontend = Frontend(backend, fp16, train)
    if backend == 'plaid':
        try:
            runner.reporter.configuration['plaid'] = importlib.import_module('plaidml').__version__
            importlib.import_module('plaidml.keras').install_backend()
        except ImportError:
            raise core.ExtrasNeeded(['plaidml-keras'])
    elif backend == 'tensorflow':
        try:
            importlib.import_module('keras.backend')
        except ImportError:
            raise core.ExtrasNeeded(['keras', 'tensorflow'])

    if fp16:
        importlib.import_module('keras.backend').set_floatx('float16')
    if train:
        runner.reporter.configuration['train'] = True
    return runner.run(frontend, networks)
