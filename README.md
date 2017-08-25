# Keras application network benchmark
Measure performance of the built-in Keras application networks: Inception-V3,
ResNet50, VGG16, VGG19, Xception, and (in Keras 2.0.6 and later) MobileNet.

To install (this will install [PlaidML](https://github.com/vertexai/plaidml):

`pip install -r requirements.txt`

To run a benchmark on a network:

`python plaidbench.py [--plaid|--no-plaid] [--train] NETWORK`

where NETWORK is one of the names "inception_v3", "resnet50", "vgg16", "vgg19",
"xception", or "mobilenet". Use --train if you want to train, otherwise the
benchmark will run inference.

By default, PlaidML is used as the backend. Specifying `--no-plaid` will utilize tensorflow,
if installed.

