# COVID-EENet
COVID-EENet: Predicting Fine-Grained Impact of COVID-19 on Local Economies

## About
- Source code and datasets of the paper COVID-EENet: Predicting Fine-Grained Impact of COVID-19 onLocal Economies.
- Since **dataset from BCCard is not open to public**, we only provide datasets except BCCard datasets.

## Installation
Requirements

- python 3.8 (Recommend Anaconda)
- TensorFlow-gpu >= 2.3.0

## Usage
- Create `tmp\checkpoint` folder.
- Go to `src` folder.
- Adjust `Config.py` by referring to the description of hyperparameters below.
- Run `python train.py` to train COVID-EENet in `src` folder.

## Hyperparameters
- `w` : prediction period, e.g., w = 14 or 28


- `h` : # multi-head, e.g, h = 4
- `e` : dimensionaligy of district-business embedding, e.g., e = 20
- `seq2seq_lstm_cell` : hidden size of LSTM, e.g., seq2seq_lstm_cell = 16
- `geographyViewFCN`  : hidden unit of FCN in Geography-View Sub-Encoder, e.g., geographyViewFCN = 1
- `epidemicViewFCN`   : hidden unit of FCN in Epidemic-View Sub-Encoder, e.g., epidemicViewFCN = 1
- `macroscopicAggFCN` : hidden unit of FCN in Macroscopic Aggregator, should be same with `w`


- `lr` : learning rate, e.g., lr = 0.001 
- `weight_decay`: weight decay of learning rate scheduler, e.g., weight_decay = 0.96
- `batch_size` : batch size, e.g., batch_size = 50
- `gpu_num` : set specific gpu, e.g., gpu_num = 0
You can adjust the hyperparameters of COVID-EENet in Config.py


