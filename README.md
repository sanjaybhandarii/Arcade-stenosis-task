# Arcade-stenosis-task
This repo is for the Runner-up solution for stenosis task for  on [ARCADE challenge](https://arcade.grand-challenge.org/) 


## Description


This project is unofficial implementation of BasicVSR. BasicVSR is designed for the task of video super-resolution. The goal of video super-resolution is to increase the resolution of a low-resolution video by generating a high-resolution counterpart. 

The BasicVSR technique involves training a deep convolutional neural network on a large dataset of low-resolution and high-resolution video pairs. During training, the network learns to use the temporal and spatial connections between frames to map low-resolution video to high-resolution video. The algorithm includes a recursive structure that allows for iterative refinement, and it's guided by four core functions: Propagation, Alignment, Aggregation, and Upsampling. BasicVSR utilizes existing components with minor modifications, resulting in improved restoration quality and speed compared to other state-of-the-art algorithms.



## Installation


1. Clone the repository:

```shell
git clone https://github.com/sanjaybhandarii/Arcade-stenosis-task
```
2.Then install torch and torchvision as:

    pip install torch torchvision

3. Install MMEngine and MMCV using MIM.

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

4. Install mmdetection
```shell
cd mmdetection
pip install -v -e .
```
    

## Usage

Change the necessary configs in train_sten.py as per your need.

Then,to train the model:

    python train_sten.py


And to infer the model on stenosis test set, use sten_inference_demo.ipynb



