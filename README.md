# DiffuTKG

## Overview

This is the code for [Predicting the Unpredictable: Uncertainty-Aware Reasoning over Temporal Knowledge Graphs via Diffusion Process](https://openreview.net/forum?id=uGHthRbN9S), accepted by Findings of ACL 2024.
![1703063577738](images/model.png)

## Requirements
```
dgl==1.1.2
dgl==1.1.2+cu117
fitlog==0.9.15
info_nce_pytorch==0.1.4
numpy==1.24.4
pandas==1.1.3
rdflib==7.0.0
scipy==1.14.0
torch==1.13.1+cu117
torch_scatter==2.0.9
tqdm==4.65.0
transformers==4.20.1
```

## Data preparation

First unzip the data files in the `data` directory, and then run `src/unseen_event.py` and `src/tri2seq.py`.

## Train and Evaluate
```
python src/main_21.py --dataset ICEWS14

python src/main_21.py --test --pattern_noise_radio 2.0 --dataset ICEWS14 --refinements_radio 1.5 --seen_addition
```

```
python src/main_21.py --dataset ICEWS18

python src/main_21.py --test --pattern_noise_radio 2.0 --dataset ICEWS18 --refinements_radio 2.0 --seen_addition
```

```
python src/main_21.py --dataset ICEWS05_15 --lr 5e-4

python src/main_21.py --test --pattern_noise_radio 2.5 --dataset ICEWS05_15 --refinements_radio 2.0 --seen_addition
```

```
python src/main_21.py --dataset GDELT --lr 5e-4

python src/main_21.py --test --pattern_noise_radio 2.5 --dataset GDELT --refinements_radio 2.0 --seen_addition
```

## Citation
```
@inproceedings{cai24predicting,
title={Predicting the Unpredictable: Uncertainty-Aware Reasoning over Temporal Knowledge Graphs via Diffusion Process},
author={Cai, Yuxiang and Liu, Qiao and Gan, Yanglei and Li, Changlin and Liu, Xueyi and Lin, Run and Yang, Jiaye},
booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
publisher = {Association for Computational Linguistics},
year={2024},
address={Bangkok, Thailand}
}
```
## Have any Questions？Email yuxiangcai at stu.uestc.edu.cn

## Acknowledge
The code of [DiffuRec](https://github.com/WHUIR/DiffuRec)
