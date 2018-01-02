#!/bin/bash

sudo apt-get update
sudo apt-get upgrade

echo "Installing essentials"
sudo apt-get install openjdk-8-jdk git python-dev python3-dev python-numpy python3-numpy build-essential python-pip python3-pip python3-venv swig python3-wheel libcurl3-dev
sudo apt-get install gcc g++ gfortran git linux-image-generic linux-headers-generic linux-source linux-image-extra-virtual libopenblas-dev

echo "Installing NVIDIA drivers"
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt-get update
sudo apt-get install -y nvidia-375 nvidia-settings

echo "Installing CUDA"
wget https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64-deb
sudo apt-get update
sudo apt-get install cuda
nvidia-smi

echo "Installing cuDNN"
wget http://developer.download.nvidia.com/compute/redist/cudnn/v6.0/cudnn-8.0-linux-x64-v6.0.tgz
tar -xzvf cudnn-8.0-linux-x64-v6.0.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*

echo "Installing TensorFlow"
sudo apt-get install libcupti-dev
pip install tensorflow-gpu
