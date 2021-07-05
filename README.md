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
# Download NAS-Bench-201
```


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
