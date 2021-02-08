# COVID-EENet
COVID-EENet: Predicting Fine-Grained Impact of COVID-19 onLocal Economies

## About
- Source code and datasets of the paper COVID-EENet: Predicting Fine-Grained Impact of COVID-19 onLocal Economies.
- Since the data from BCCard is not open to public, we only provide datasets except BCCard datasets.

## Installation
Requirements

- python 3.8 (Recommend Anaconda)
- TensorFlow-gpu >= 2.3.0

## Usage
- Run "python train.py" to train COVID-EENet

## Hyperparameters
- `h` : # multi-head, e.g, h = 4
- `e` : District-business embedding dimension, e.g., e = 20

- `lr` : learning rate, e.g., lr = 0.001 
- `weight_decay`: weight decay of learning rate scheduler, e.g., weight_decay = 0.96
- `batch_size` : batch size, e.g., batch_size = 50 
- `w` : prediction period, e.g., w = 14 or 28

You can adjust the hyperparameters of COVID-EENet in Config.py


