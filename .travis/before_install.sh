#!/bin/bash -e

mkdir -p $HOME/.cache/eggs
ln -s $HOME/.cache/eggs .eggs

# Install dependencies.
if [ "$TRAVIS_OS_NAME" == "linux" ]; then
  # TODO: Once Travis supports an Ubuntu distribution more recent than Trusty,
  # we should install the distribution's libprotoc-dev and protobuf-compiler,
  # instead of compiling it outselves.
  pb_dir="$HOME/.cache/pb"
  mkdir -p $pb_dir
  wget -qO- "https://github.com/google/protobuf/releases/download/v$PB_VERSION/protobuf-$PB_VERSION.tar.gz" | tar -xz -C $pb_dir --strip-components 1
  cd $pb_dir && ./configure && make && make check && sudo make install && sudo ldconfig && cd -
  
  # TODO: Consider using beignet, since it can be installed from apt.
  ocl_dir="$HOME/.cache/ocl"
  mkdir -p "$ocl_dir"
  wget -qO- https://storage.googleapis.com/external_build_repo/opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25.tgz | tar -xz -C $ocl_dir --strip-components 1
  silent_file="$(pwd)/.intel-ocl-silent"
  cd $ocl_dir && sudo ./install.sh --cli-mode --silent $silent_file && cd -

elif [ "$TRAVIS_OS_NAME" == "osx" ]; then
  if [ "$PYTHON" == "python2" ]; then
    brew install python@2 || brew link --overwrite python@2
  elif [ "$PYTHON" == "python3" ]; then
    brew upgrade python
  fi
  brew install protobuf
else
  echo Unknown OS: $TRAVIS_OS_NAME
  exit 1
fi
