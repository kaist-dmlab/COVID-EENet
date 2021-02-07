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
- `h` : # of multi-head
- `e` : District-business embedding dimension 

- `lr` : learning rate 
- `weight_decay`: weight decay of learning rate scheduler
- `batch_size` : batch size 

You can adjust the hyperparameters of COVID-EENet in Config.py


