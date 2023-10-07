# FairFor

Source code for the paper,
["Learning Informative Representation for Fairness-aware Multivariate Time-series Forecasting: A Group-based Perspective"](https://arxiv.org/abs/2301.11535),
accepted by TKDE.

## Overview
In this work, we formulate the MTS fairness modeling problem as learning informative representations attending to both advantaged and disadvantaged variables. 
Accordingly, we propose a novel framework, named FairFor, for fairness-aware MTS forecasting, i.e., fair MTS forecasting.

## Requirements
- Python 3.6
- matplotlib == 3.3.4
- numpy == 1.19.5
- pandas == 1.1.5
- scikit_learn == 0.24.2
- torch == 1.8.0

## Datasets
- PeMSD7(M) - https://dot.ca.gov/programs/traffic-operations/mpr/pems-source
- Solar-Energy - http://www.nrel.gov/grid/solar-power-data.html
- Traffic - https://archive.ics.uci.edu/ml/datasets/PEMS-SF
- ECG5000 - http://www.timeseriesclassification.com/description.php?Dataset=ECG5000

## Baselines
- LSTNet - https://github.com/laiguokun/LSTNet
- TPA-LSTM - https://github.com/shunyaoshih/TPA-LSTM
- TS2VEC - https://github.com/yuezhihan/ts2vec
- Informer - https://github.com/zhouhaoyi/Informer2020
- Pyraformer - https://github.com/ant-research/Pyraformer
- MTGNN - https://github.com/nnzhan/MTGNN
- StemGNN - https://github.com/microsoft/StemGNN
- AGCRN - https://github.com/LeiBAI/AGCRN

## Citation
If you find our work useful, please consider citing the following paper

```text
@article{DBLP:journals/corr/abs-2301-11535,
  author       = {Hui He and
                  Qi Zhang and
                  Shoujin Wang and
                  Kun Yi and
                  Zhendong Niu and
                  Longbing Cao},
  title        = {Learning Informative Representation for Fairness-aware Multivariate
                  Time-series Forecasting: {A} Group-based Perspective},
  journal      = {CoRR},
  volume       = {abs/2301.11535},
  year         = {2023}
}
```
