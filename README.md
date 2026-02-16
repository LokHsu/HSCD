# HSCD

This repository contains source codes and datasets for the paper:

- Hierarchical Spurious Correlation Disentanglement for Multi-Behavior Recommendation

## Usage
### Train & Test

Run the following script for preprocessing:
```shell
python ./data/preprocess.py
```

- Train HSCD on the `Taobao` dataset
```shell
python ./src/main.py --dataset taobao --lr 5e-4 --lam 1.5
```

- Train HSCD on the `Tmall` dataset
```shell
python ./src/main.py --dataset tmall  --lr 5e-4  --lam 2
```

- Train HSCD on the `Retail` dataset
```shell
python ./src/main.py --dataset retailrocket  --lr 5e-4 --lam 2.5
```
