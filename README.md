# Stronger NAS with Weaker Predictors
Code used for [Stronger NAS with Weaker Predictors](https://arxiv.org/abs/2102.10490). 

## Main Pipeline
![An illustration of our Weak Predictors Pipeline](assets/process.png)


![Visualization of the Search Dynamics](assets/dynamics.png)

## Results

### Environment
```bash
pip install -r requirements.txt
```

### NASBenchs Search Space

- NAS-Bench-101
```bash
# Download NAS-Bench-101
```
- NAS-Bench-201
```bash
# Download NAS-Bench-201
```

### Open Domain Search Space

- ImageNet (MobileNet Setting)
```bash
cd pytorch-image-models;
# 800 Samples model
bash distributed_train.sh $NUM_GPU $IMAGENET_PATH --model ofa_mbv3_800 -b 128 \
--sched cosine --img-size 236 --epochs 300 --warmup-epochs 3 --decay-rate .97 \
--opt rmsproptf --opt-eps .001 -j 10 --warmup-lr 1e-6 --weight-decay 1e-05 --drop 0.3 \
--drop-path 0.0 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 \
--remode pixel --reprob 0.2 --lr 1e-02 --output $LOG_PATH \
--experiment res_224/bs_128/cosine/lr_5e-03/wd_1e-05/epoch_300/dp_0.0 --log-interval 200
```
```bash
cd pytorch-image-models;
# 1000 Samples model
bash distributed_train.sh $NUM_GPU $IMAGENET_PATH --model ofa_mbv3_1000 -b 128 \
--sched cosine --img-size 236 --epochs 600 --warmup-epochs 3 --decay-rate .97 \
--opt rmsproptf --opt-eps .001 -j 10 --warmup-lr 1e-6 --weight-decay 1e-05 --drop 0.3 \
--drop-path 0.0 --model-ema --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 \
--remode pixel --reprob 0.2 --lr 1e-02 --output $LOG_PATH \
--experiment res_224/bs_128/cosine/lr_5e-03/wd_1e-05/epoch_600/dp_0.0 --log-interval 200
```
Tensorboard.dev Logs: [Links](https://tensorboard.dev/experiment/YuDEyzRSQpOQT7ZEZa8tNg/#scalars)

## Acknowledgement
NASBench Codebase from [AutoDL-Projects](https://github.com/D-X-Y/AutoDL-Projects)  
ImageNet Codebase from [timm](https://github.com/rwightman/pytorch-image-models)

## Citation
if you find this repo is helpful, please cite
```
@article{wu2021weak,
  title={Stronger NAS with Weaker Predictors},
  author={Junru Wu and Xiyang Dai and Dongdong Chen and Yinpeng Chen and Mengchen Liu and Ye Yu and Zhangyang Wang and Zicheng Liu and Mei Chen and Lu Yuan},
  journal={arXiv preprint arXiv:2102.10490},
  year={2021}
}
```
