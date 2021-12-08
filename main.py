import sys
import os
import math
import copy
import argparse

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn

from Config import Config
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

from dataset.dataset import CovidDataset, collate_fn
from models.covideenet import weights_init_classifier
from model_utils import check_perf_five_models,load_perf_model,perf,save_model_prediction
from model_utils import save_results_district_buz_pair,get_rmse_mean_std

parser = argparse.ArgumentParser(description='COVIDEENet')

parser.add_argument('--model_name', type=str, default='covideenet',
                    help='type one of the comparing algorithms including COVID-EENet')
parser.add_argument('--fname', type=str, default='D14_model_{}.pt',
                   help='type the file name of the parameters, predictions, experiment results')
parser.add_argument('--pred_len', type=int, default=14,
                    help='type the predicting length of algorithms')
parser.add_argument('--cuda', type=int, default=0,
                    help='type the number of gpu')
parser.add_argument('--train', action='store_true',
                    help='type when you train the model')
parser.add_argument('--test', action='store_true',
                    help='type when you validate the model')
parser.add_argument('--save_prediction', action='store_true',
                    help='type when you save the predictions of the algorithms')
parser.add_argument('--save_metric_result', action='store_true',
                    help='type when you save the experiment results of the algorithms')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)

def main(model, trainloader, testloader, lossmask, patience=10, mean=None, std=None):
    
    EPOCHS = 100
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    model.to(torch.float32)
#     model.apply(weights_init_classifier)
    
    best_val_rmse = float('inf')
    val_rmse_list = []
    lossmask = lossmask.to(model.config.device)
    
    
    for e in range(EPOCHS):
        # train
        model.train()
        total_mseloss = 0.
        for batch_x in trainloader:
            
            mseloss, _, modeling_output = model(batch_x,) # mseloss: (bs, #ind, #pred)
            rep = torch.div(mseloss.size(0), model.region, rounding_mode='floor')
            mseloss = (mseloss * (lossmask.repeat(rep,1).unsqueeze(-1)))
            mseloss = mseloss.sum()/(lossmask.sum()*(rep))
            rmseloss = torch.sqrt(mseloss)

            optimizer.zero_grad()
            rmseloss.backward()
            total_mseloss += rmseloss.item()
            optimizer.step()
            
        ## Start validation ###################################################
        model.eval()
        total_rmseloss = 0.
        
        for val_x in testloader: # valloader,testloader
            with torch.no_grad():
                mseloss, y_hat, modeling_output = model(val_x,)
                mseloss = mseloss.cpu()
                rep = torch.div(mseloss.size(0),model.region, rounding_mode='floor')
                mseloss = (mseloss*lossmask.repeat(rep,1).unsqueeze(-1).cpu()).mean()
                rmseloss = torch.sqrt(mseloss)
                total_rmseloss += rmseloss
                
                if config.model_name == 'covideenet':
                    y = val_x[1]
                else: 
                    y = val_x[-1]
                maeloss = nn.L1Loss(reduction='none')(y_hat, 
                                                      y.to(model.config.device))
                maeloss = maeloss.cpu()
                maeloss = (maeloss*lossmask.repeat(rep,1).unsqueeze(-1).cpu()).mean()

        
        val_rmse_list.append(total_rmseloss)
        if total_rmseloss < best_val_rmse:
            print("BEST VAL- Epoch : {}, validation loss : {}".format((e+1),total_rmseloss))
            best_val_rmse = total_rmseloss
            best_model_state_dict = copy.deepcopy(model.state_dict())
            earlystop = 0
        else : 
            print("Epoch : {}, validation loss : {}".format((e+1),total_rmseloss))
            earlystop += 1
        
        if earlystop >= patience:
            print("earlystoped!! best_epoch was {}\n\n".format((e+1)- patience))
            
            return best_model_state_dict, val_rmse_list[(e)-patience]

config = Config(
    p = args.pred_len,
    datapath = "None"
)

config.model_name = args.model_name
    
print("Loading data...")
if config.model_name == "covideenet":
    if config.p == 14:
        trainloader, testloader = torch.load("data/trainloader_testloader_14days.pt")
        covid_mask = torch.load("data/covid_mask_14days.pt")
        lossmask = torch.load("data/lossmask_district_buz_14days.pt")
    else:
        trainloader, testloader = torch.load("data/trainloader_testloader.pt")
        covid_mask = torch.load("data/covid_mask.pt")
        lossmask = torch.load("data/lossmask_district_buz.pt")

    config.covid_start = trainloader.dataset.covid_start  if trainloader.dataset.covid_start else None
    
else: 
    config.dnnfeat_dim = 3810
    config.lr = 1e-3
    if config.model_name == "defsi": # for defsi
        config.seasonal_week = 14
        
    if config.p == 14:
        if args.train:
            trainloader = torch.load("data/trainloader_D14.pt")
        testloader = torch.load("data/testloader_D14.pt")
        covid_mask = torch.load("data/covid_mask_14days.pt")
        lossmask = torch.load("data/lossmask_district_buz_14days.pt")
    else:
        if args.train:
            trainloader = torch.load("data/trainloader_D28.pt")
        testloader = torch.load("data/testloader_D28.pt")
        covid_mask = torch.load("data/covid_mask.pt")
        lossmask = torch.load("data/lossmask_district_buz.pt")

device = torch.device("cuda")
dtype = torch.float32
config.device = device
config.dtype = dtype

root_directory = os.getcwd()
directory = root_directory+"/models_state_dict/"

industry_list = torch.load("data/industry_list.pt")
city_dict = torch.load("data/city_dict.pt")
mass_inf_info = torch.load("data/mass_inf_info.pt")

if config.model_name == "covideenet":
    from models.covideenet import *
elif config.model_name == "seq2seqattn":
    from models.seq2seqattn import *
elif config.model_name == "defsi":
    from models.defsi import *
elif config.model_name == "tada":
    from models.tada import *
elif config.model_name == "tcn":
    from models.tcn import *

if __name__ == "__main__":
    fname = args.fname
    if args.train :
        print("Start training...")
        if not os.path.isdir(directory):
            os.mkdir(directory)
        else:
            print("{} directory already exists.".format(directory))
        
        for i in range(5):
            if config.model_name == "covideenet":
                model = COVIDEENet(config).to(device)
            elif config.model_name == "seq2seqattn":
                model = Seq2SeqATTN(config).to(device)
            elif config.model_name == "defsi":
                model = DEFSI(config).to(device)
            elif config.model_name == "tada":
                model = TADA(config).to(device)
            elif config.model_name == "tcn":
                model = TCN_FCN(config).to(device)

            best_model_state_dict, val_loss = main(model, 
                                                   trainloader, testloader,
                                                   lossmask, patience=20, mean=None, std=None)

            torch.save(obj = best_model_state_dict,
                       f = directory+fname.format(i))
        print("DONE\n")

    else: 
        print("Skip training...")
        
    if args.test :
        print("Start testing on test data...")
        dir_arg = directory        
        covideenet_best, val_x = check_perf_five_models(model_name=config.model_name,
                                                        testloader=testloader,
                                                        directory=dir_arg,
                                                        model_state_dict_fname=fname,
                                                        config=config)
        print("DONE\n")
    else:
        print("Skip testing...")
        
    if args.save_prediction:
        print("Start saving model predictions on test data...")
        dir_arg = directory
        RESULT_SAVE_DIR ="model_prediction"
        
        if not os.path.isdir(os.path.join(os.getcwd(),RESULT_SAVE_DIR)):
            os.mkdir(os.path.join(os.getcwd(),RESULT_SAVE_DIR))
        else:
            print("{} directory already exists.".format(os.getcwd(),RESULT_SAVE_DIR))
            
        save_model_prediction(model_name=config.model_name, 
                              directory=dir_arg,
                              model_state_dict_fname=fname,
                              config=config, 
                              result_save_directory=RESULT_SAVE_DIR, 
                              testloader=testloader)
        print("DONE\n")
    else:
        print("Skip saving predictions...")
        
    if args.save_metric_result:
        print("Start calculating metric results on test data...")
        RMSE_SAVE_DIR = "RMSE_district_buz_pairs"
        dir_arg = directory
        
        if not os.path.isdir(os.path.join(os.getcwd(),RMSE_SAVE_DIR)):
            os.mkdir(os.path.join(os.getcwd(),RMSE_SAVE_DIR))
        else:
            print("{} directory already exists.".format(os.getcwd(),
                                                        RMSE_SAVE_DIR))
        
        get_rmse_mean_std(config,
                          model_name=config.model_name,
                          testloader=testloader,
                          model_directory=dir_arg,
                          result_save_directory=RMSE_SAVE_DIR, 
                          model_state_dict_fname=fname,
                          industry_list=industry_list, 
                          city_list=city_dict)
        print("DONE\n")
    else:
        print("Skip saving metric results...")
        
        