#!/usr/bin/env bash

python -V
nvcc -V
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install Cython termcolor numpy thop matplotlib tqdm tensorboard==2.4.0 scipy sklearn xgboost shap lightgbm ipython
ls DATASET
