# COVID-EENet
COVID-EENet: Predicting Fine-Grained Impact of COVID-19 on Local Economies

## About
- Source code and datasets of the paper COVID-EENet: Predicting Fine-Grained Impact of COVID-19 onLocal Economies, AAAI 2022.
- Since **dataset from BCCard is not open to public**, we only provide epidemic-view feature and the physical distance dataset of geography-view feature.


## Installation
Requirements
- Ubuntu 16.04.7 LTS
- python 3.8 (Recommend Anaconda)
- Pytorch >= 1.9.0

## Usage
- Run `python main.py` to train COVID-EENet
```bash
python main.py -h
usage: main.py [-h] [--model_name MODEL_NAME] [--fname FNAME] [--pred_len PRED_LEN] [--cuda CUDA] [--train] [--test] [--save_prediction] [--save_metric_result]
COVIDEENet
optional arguments:
  -h, --help                 show this help message and exit
  --model_name MODEL_NAME    type one of the comparing algorithms including COVID-EENet
  --fname FNAME              type the file name of the parameters, predictions, experiment results
  --pred_len PRED_LEN        type the predicting length of algorithms
  --cuda CUDA                type the number of gpu
  --train                    type when you train the model
  --test                     type when you validate the model
  --save_prediction          type when you save the predictions of the algorithms
  --save_metric_result       type when you save the experiment results of the algorithms
```
### After training the model, you can find 
- the learned parameters in directory `models_state_dict`
- the predictions in directory `model_prediction`
- the experiment results in directory `RMSE_district_buz_pairs`

## Hyperparameters
Please check the hyperparameters of COVID-EENet defined in Config.py and supplementary material.



