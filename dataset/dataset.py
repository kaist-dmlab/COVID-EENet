from torch.utils.data import Dataset
import torch
import numpy as np

class CovidDataset(Dataset):
    def __init__(self, X, y, lossMask, metadata, 
                 covid_elapsed_dt, normed_severity, 
                 covid_start, prediction_len):
        # X : (N, #massinf, #feats)
        X = X.transpose((2,1,0)) # (#feats, #massinf, N)
        self.business_structure_target, \
        self.business_structure_infected, \
        self.customer_structure_target, \
        self.customer_structure_infected, \
        self.index_target_idx, \
        self.index_infected_idx, \
        self.physical_distance, \
        self.contextual_distance, \
        self.covid_outbreak_business, \
        _,\
        _, \
        self.weekdays, \
        self.mask, \
        self.covidMask = [torch.from_numpy(np.array(X[i].tolist())).to(torch.float32)
                          for i in range(X.shape[0])] # (#massinf, N, dim*)
                                           
        self.y = torch.from_numpy(y).to(torch.float32)
        self.loss_mask = torch.from_numpy(lossMask).to(torch.float32)
        self.metadata = metadata
        
        self.covid_elapsed_day = covid_elapsed_dt
        self.epidemiological_severity = normed_severity # (len=305+28, #massinf, n_feats=6,)
        
        self.covid_start = covid_start
        self.prediction_len = prediction_len
        self.len = self.y.size(0)
        
    def __getitem__(self, index):
        #covid_elapsed_day (:index//25, #massinf, dim=6)

        valid_len = self.mask[:,index,:].sum(1) # (#massinf, len)
        valid_len = valid_len[0].to(torch.long)
    #                 self.covidMask[:,index,:], \
        return self.business_structure_target[:,index,:], \
                self.business_structure_infected[:,index,:], \
                self.customer_structure_target[:,index,:], \
                self.customer_structure_infected[:,index,:], \
                self.index_target_idx[:,index,:], \
                self.index_infected_idx[:,index,:], \
                self.physical_distance[:,index,:], \
                self.contextual_distance[:,index,:], \
                self.covid_outbreak_business[:,index,:], \
                self.epidemiological_severity[:valid_len+self.prediction_len,],\
                self.covid_elapsed_day[:valid_len+self.prediction_len,:], \
                self.weekdays[:,index,:], \
                self.mask[:,index,:], \
                self.covidMask[:,index], \
                self.y[index], \
                self.loss_mask[index], \
                self.metadata[index], \

    def __len__(self):
        return self.len
    
def collate_fn(samples):
    n_inp = len(samples[0])  
    samples = [feat for sample in samples for feat in sample]
    
    samples_tuple = ()
    for i in range(n_inp - 3): # no y, lossmask, metadata
        if (i == 9) or (i == 10): # variable length features
            padded_feats = torch.nn.utils.rnn.pad_sequence(samples[i::n_inp], batch_first=True, )
            samples_tuple += (padded_feats,)
            continue
        samples_tuple += (torch.stack(samples[i::n_inp], dim=0),)
        
    return samples_tuple,\
            torch.stack(samples[n_inp-3::n_inp], dim=0), \
            torch.stack(samples[n_inp-2::n_inp], dim=0), \
            np.stack(samples[n_inp-1::n_inp], axis=0)
