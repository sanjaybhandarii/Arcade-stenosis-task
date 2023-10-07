# Arcade-stenosis-task
This repo is for the Runner-up solution for stenosis task for  on [ARCADE challenge](https://arcade.grand-challenge.org) 


## Description



## Installation


1. Clone the repository:

```shell
git clone https://github.com/sanjaybhandarii/Arcade-stenosis-task
```

2. Then install torch and torchvision as:

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



