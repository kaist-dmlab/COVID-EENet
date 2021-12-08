import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from models.TCN.TCN.copy_memory.model import TCN

class TCN_FCN(nn.Module):
    def __init__(self, config, 
                 mass_inf = 30, 
                 num_regions=25, 
                 hidden_size=50, 
                 tcn_layer=8, 
                 kernel_sz=8,
                 dropout_p=0.1):
        super(TCN_FCN, self).__init__()
        
        self.config = config
        self.region = config.r
        self.numOfMassInfection = mass_inf
        self.numOfIndustry = 34
        self.severity_emb = 6
        self.hidden_size = hidden_size
        
        input_size = 1 + self.severity_emb + self.hidden_size # 41
        self.severity_TCN = TCN(input_size=input_size, 
                                 output_size=hidden_size,
                                 num_channels=[hidden_size]*tcn_layer, 
                                 kernel_size=kernel_sz, 
                                 dropout=dropout_p)
        # (bs, D, T) -> (bs, T, D)
        
        self.concat_linear = nn.Linear(self.hidden_size*2, self.hidden_size) 
        self.tanh = nn.Tanh()
        # MSE loss for severity
        self.severity_linear = nn.Linear(in_features= self.hidden_size, out_features= self.severity_emb)
        
        self.linear_dnnfeatures = nn.Linear(self.config.dnnfeat_dim, self.hidden_size)
        self.week_emb_linear = nn.Linear(self.hidden_size, 1)
        self.linear_y_hat = nn.Linear(hidden_size,self.numOfIndustry)
        self.last_linear = nn.Linear(config.p,config.p)
        
        self.embedding = nn.Embedding(7, self.hidden_size)
        self.loss = torch.nn.MSELoss(reduction="none")
    
    def int_lstm_dt(self, data_int, input_len):
         # return output_int
        

        data_int_shape = data_int.size() # (batch_size = #dates*#gus, 362=#dates, 30=#mass_inf_cases)
        data_int = data_int.transpose(1,2).contiguous().view(data_int_shape[0]*data_int_shape[2], data_int_shape[1], 1)
        # data_int: (bs*mass, T=362, 1)

        input_len_int_syn = input_len.unsqueeze(-1).expand(input_len.size(0),data_int_shape[2]).contiguous().view(-1)
        # (bs*mass)
        if input_len[0] > 50:
            input_len__50 = torch.tensor([50]*data_int.size(0))
            packed_input_int = torch.nn.utils.rnn.pack_padded_sequence(data_int[:,int(input_len[-1])-50:int(input_len[-1]),:], 
                                                                       input_len__50.tolist(),
                                                                       batch_first=True,
                                                                       enforce_sorted=False
                                                                      )

        else:
            packed_input_int = torch.nn.utils.rnn.pack_padded_sequence(data_int[:,:int(input_len[-1]),:],
                                                                       input_len_int_syn.tolist(),
                                                                       batch_first=True,
                                                                       enforce_sorted=False
                                                                        )
            

            
        input_int_dec = data_int[:,-self.config.p:,:] # (bs*mass, pred_len, 1)
        
        return packed_input_int, input_len_int_syn, input_int_dec

    def ext_lstm_dt(self, final_week, input_len_int_syn, final_dnn) :
        # return output_ext
        final_week = self.embedding(final_week) # (bs, config.p, dim)
        
        bs = final_week.size(0)
        T = input_len_int_syn[-1].to(torch.long)
        
        
        input_ext_dec = final_week.unsqueeze(1).expand(bs, 30, self.config.p, self.hidden_size)
        input_ext_dec = input_ext_dec.reshape(-1, self.config.p, self.hidden_size) # (bs*mass, pred_len, hidden_size)
        
        if T>50:
            input_len__50_ext = torch.tensor([50]*(self.batch_sz*30))
            final_week = torch.cat([final_week for i in range(T//self.config.p+1)], 
                                       axis=1)[:,-50:,:]
            final_week = final_week.squeeze()
            final_week = final_week.unsqueeze(1).expand(bs, 30, final_week.size(-2), self.hidden_size).contiguous()
            final_week = final_week.view(-1,final_week.size(-2), self.hidden_size) # (bs*mass, len, self.hidden_size)
            packed_input_ext = torch.nn.utils.rnn.pack_padded_sequence(final_week, 
                                                                       input_len__50_ext.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)
        else :
            final_week = torch.cat([final_week for i in range(T//self.config.p+1)], 
                                       axis=1)[:,-T:,:]
            final_week = final_week.squeeze()
            final_week = final_week.unsqueeze(1).expand(bs, 30, final_week.size(-2), self.hidden_size).contiguous()
            final_week = final_week.view(-1,final_week.size(-2), self.hidden_size) # (bs*mass, len, self.hidden_size)
            packed_input_ext = torch.nn.utils.rnn.pack_padded_sequence(final_week, 
                                                                       input_len_int_syn.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)

        return packed_input_ext, input_ext_dec


    def syn_TCN(self, 
                data_syn,
                final_dnn,
                input_lens,
                packed_inputs, 
                dec_inp ):
        
        packed_input_int, packed_input_ext = packed_inputs
        input_len, input_len_int_syn = input_lens
        weekdays_dec = dec_inp
        # return output_syn
        cut_t = input_len[-1].to(torch.long)
        T = input_len_int_syn[-1].to(torch.long)
        bs = data_syn.size(0)
        
        # get severity_dec
        severity_dec = torch.cat((data_syn[:,[cut_t],:], data_syn[:,-self.config.p:,:]), dim=1)
        severity_dec = severity_dec.view(bs, self.config.p+1, self.config.c, -1)
        severity_dec = severity_dec.transpose(1,2).contiguous().view(bs*self.config.c, self.config.p+1, -1)
        # severity_dec: (bs*#mass, self.config.p+1, dim)
        
        if T>50:
            data_syn = data_syn[:,cut_t-50:cut_t,:]
            data_syn_shape = data_syn.size()
        else : 
            data_syn = data_syn[:,:cut_t,:]
            data_syn_shape = data_syn.size()
        
        data_syn = data_syn.view(data_syn_shape[0], data_syn_shape[1], self.config.c, -1)
        severity = data_syn.transpose(1,2).contiguous().view(data_syn.size(0)*self.config.c, data_syn_shape[1], -1)
        # severity: (bs*#mass, len, dim)
        
        # pack input
        elapsed, elapsed_len = torch.nn.utils.rnn.pad_packed_sequence(packed_input_int, batch_first=True) 
        weekday, weekday_len = torch.nn.utils.rnn.pad_packed_sequence(packed_input_ext, batch_first=True) 
        
        data_cat = torch.cat([severity, elapsed, weekday], axis=-1)
        # data_cat: (bs*mass, T, 41 = sev_emb(6)+week_emb(self.hidden_size)+elapsed(1))
        
        if T>50:
            input_len__50 = torch.tensor([50]*data_cat.size(0))
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(data_cat, 
                                                                       input_len__50.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)
        else :
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(data_cat, 
                                                                       input_len_int_syn.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)
            
        # input packed_input & output packed_output 
        data_cat = data_cat.transpose(1,2).to(self.config.dtype) 
        severity_tcn = self.severity_TCN(data_cat) # (bs*mass, D, T) -> (bs*mass, T=50, hidden)
        severity_tcn = severity_tcn[:, -self.config.p:, :] # (bs*mass, pred_len, hidden)
        
        
        # add dnn features
        dnnfeat = self.linear_dnnfeatures(final_dnn.to(torch.float32)) # (bs, mass, hidden)
        dnnfeat = dnnfeat.view(-1, self.hidden_size).unsqueeze(1) # dnnfeat:(bs*mass,1,hidden)
        
        # add week features
        week_emb = self.week_emb_linear(weekdays_dec).squeeze() # (bs*#mass, pred_len, 1) 
        week_emb = week_emb.unsqueeze(-1).expand(-1,-1, self.hidden_size) # (bs*#mass, pred_len, ind) 
        
        severity_tcn = severity_tcn + dnnfeat + week_emb # severity_tcn: (bs*mass,pred_len,hidden)
        
        return severity_tcn
    
    def compute_y_hat(self, severity_tcn, covid_mask, modeling_output=None):
        # severity_tcn: (bs*mass,pred_len,hidden)
        y_hat = self.tanh(self.linear_y_hat(severity_tcn))
        y_hat = y_hat.view(self.batch_sz, 30, self.config.p, -1) # y_hat (bs, #massinf, pred_len, hid)
        y_hat = y_hat * covid_mask.unsqueeze(-1).unsqueeze(-1)
        if modeling_output :
            modeling_output["y_hat_bef_mean"] = y_hat.cpu()
            
        y_hat = y_hat.sum(dim=1) # (bs, config.p, hid)
        y_hat = y_hat / covid_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
        if modeling_output :
            modeling_output["y_hat"] = y_hat.cpu()

        return y_hat if modeling_output is None else (y_hat, modeling_output)
    
    def forward(self, x, verbose=False, inspect=False):
        
        if inspect: 
            modeling_output = {}
            
        final_dnn, data_int, data_syn, final_week, final_mask, covid_mask, y_train = x
        final_dnn = final_dnn.to(self.config.device, )
        data_int = data_int.to(self.config.device, )
        data_syn = data_syn.to(self.config.device, )
        final_week = final_week.to(self.config.device, )
        final_mask = final_mask.to(self.config.device, )
        covid_mask = covid_mask.to(self.config.device, self.config.dtype)
        
        y_train = y_train.to(self.config.device)
        
        self.batch_sz = data_int.size(0) 
        input_len = final_mask.sum(axis=1)
        
        # int_lstm ######################### : output_int
        packed_input_int, input_len_int_syn, input_int_dec = self.int_lstm_dt(data_int, input_len, ) # return output_int_data

        # ext_lstm ######################### : output_ext
        packed_input_ext, input_ext_dec = self.ext_lstm_dt(final_week, input_len_int_syn, final_dnn,) # return output_ext_data
        
        # syn_lstm ######################### : output_syn
        severity_tcn = self.syn_TCN(data_syn, final_dnn, 
                                    input_lens=(input_len,input_len_int_syn),
                                    packed_inputs=(packed_input_int, packed_input_ext),
                                    dec_inp=input_ext_dec)
        # return severity_tcn (bs*#massinf, pred_len, hidden_size)
    
        if inspect:
            logits, modeling_output = self.compute_y_hat(severity_tcn, covid_mask, modeling_output) # return logits, which is y_hat
            # (bs, pred, ind)
            
        else :
            logits = self.compute_y_hat(severity_tcn, covid_mask,) # return logits = y_hat
            
        logits = logits.transpose(1,2).contiguous() # (bs, ind, pred)
        logits = self.last_linear(logits)
        # compute MSELoss ######################### logits: (bs, config.p, hid) 
        return (self.loss(logits, y_train.to(torch.float32)), logits, modeling_output) \
                if inspect \
                else (self.loss(logits, y_train.to(torch.float32)), logits, None)
    
    