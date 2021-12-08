import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class DEFSI(nn.Module):
    def __init__(self, config, mass_inf = 30, num_regions=25, hidden_size=50, rnn_layer=1, dropout_p=0.1):
        super(DEFSI, self).__init__()
        
        self.config = config
        self.region = config.r
        self.sw = config.seasonal_week # or l in the paper DEFSI
        
        self.numOfMassInfection = mass_inf
        self.numOfIndustry = 34
        self.severity_emb = 6
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(7, self.hidden_size)
        
        self.data_emb_linear = nn.Linear(hidden_size+self.severity_emb+1,hidden_size,)
        self.lstm_within_season = nn.LSTM(input_size=hidden_size,#41
                                            hidden_size=hidden_size//2,
                                            num_layers=1,
                                            bidirectional=True,
                                            batch_first=True)
        self.linear_within_season = nn.Linear(in_features=self.hidden_size,
                                              out_features=self.hidden_size)
        self.lstm_between_season = nn.LSTM(input_size=hidden_size,#41
                                            hidden_size=hidden_size//2,
                                            num_layers=1,
                                            bidirectional=True,
                                            batch_first=True)
        self.linear_between_season = nn.Linear(in_features=self.hidden_size,
                                              out_features=self.hidden_size)
        
        
        self.linear_concat = nn.Linear(self.hidden_size*2, self.hidden_size) 
        
        # MSE loss for severity
        self.linear_severity = nn.Linear(in_features= self.hidden_size*2, out_features= self.severity_emb)

        self.linear_dnnfeatures = nn.Linear(self.config.dnnfeat_dim, self.hidden_size)
        self.week_emb_linear = nn.Linear(self.hidden_size, 1)
        
        self.cat_linear = nn.Linear(3,1) 
        self.final_linear = nn.Linear(self.hidden_size, self.numOfIndustry) 
        self.tanh = nn.Tanh()
        
        ##### Loss Function ############
        self.loss = torch.nn.MSELoss(reduction="none")
        
    def int_lstm_dt(self, data_int, input_len, step):
        """
        @step: \in [0, config.p]
        """

        data_int_shape = data_int.size() # (batch_size = #dates*#gus, 362=#dates, 30=#mass_inf_cases)
        data_int = data_int.transpose(1,2).contiguous().view(data_int_shape[0]*data_int_shape[2], data_int_shape[1], 1)
        # data_int: (bs*mass, T=362, 1)
        bs_mass = data_int.size(0)
        
        input_int_dec = data_int[:,-self.config.p:,:] # (bs*mass, pred_len, 1)
        input_int_dec = input_int_dec[:,:step,:] # (bs*mass, step, 1)
        
        input_len_int_syn = input_len.unsqueeze(-1).expand(input_len.size(0),data_int_shape[2]).contiguous().view(-1)
        # input_len_int_syn: (bs*mass)
        
        uniq_val, inv_ind = input_len_int_syn.unique(return_inverse=True)
        uniq_val = uniq_val.to(torch.long)
        uniq_val_dict = {_: torch.arange(uniq_val[_]-50, uniq_val[_]) for _ in range(uniq_val.size(0))}
        gather_ind = torch.stack([uniq_val_dict[l.item()] for l in inv_ind], 
                                 dim=0).to(torch.long) # (bs*mass, 50)
        if step > 0:
            gather_ind_for_multipred = [gather_ind[:,[-1]] + i for i in range(step)]
            gather_ind_for_multipred = torch.cat(gather_ind_for_multipred, dim=-1) # (bs*mass, step)
            gather_ind_all = torch.cat((gather_ind,gather_ind_for_multipred), dim=-1) 
            # gather_ind: (bs*mass, 50+step)
            data_int[torch.arange(bs_mass).unsqueeze(-1), gather_ind_for_multipred] = input_int_dec
        else :
            gather_ind_all = gather_ind
            gather_ind_for_multipred = None
            
        input_len_int_syn = input_len_int_syn + step
        
        if input_len[0] > 50:
            input_len__50 = torch.tensor([50]*data_int.size(0)) + step
            packed_input_int = torch.nn.utils.rnn.pack_padded_sequence(data_int[torch.arange(bs_mass).unsqueeze(-1),
                                                                                gather_ind_all], 
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
    
        uniq_val, inv_ind = input_len_int_syn.unique(return_inverse=True)
        uniq_val_dict_bet_season = {_: torch.flip(torch.arange(uniq_val[_]-1, -1, -self.sw), dims=[0]) 
                                    for _ in range(uniq_val.size(0))}
        gather_ind_bet_season = torch.nn.utils.rnn.pad_sequence([uniq_val_dict_bet_season[l.item()] for l in inv_ind],
                                                     batch_first = True,
                                                     padding_value = -1.).to(torch.long) 
        # gather_ind: (bs*mass, self.max_validlen // self.sw)
        
        input_len_int_syn_between_season = input_len_int_syn / self.sw # (bs*mass)
        input_len_int_syn_between_season_trunc = input_len_int_syn_between_season.to(torch.int)
        ceil_mask = (input_len_int_syn_between_season - input_len_int_syn_between_season_trunc) < 1e-4
        input_len_int_syn_between_season = torch.ceil(input_len_int_syn_between_season).to(torch.int)
        input_len_int_syn_between_season[ceil_mask] = input_len_int_syn_between_season_trunc[ceil_mask]
        
        
        # slice until max length (int(input_len[-1])) and give the valid len (input_len_int_syn)
        packed_input_int_between_season = data_int[:,:int(input_len_int_syn[-1]),:] # (bs*mass, max_validlen, feats)
        packed_input_int_between_season = packed_input_int_between_season[torch.arange(bs_mass).unsqueeze(-1),
                                                                          gather_ind_bet_season]

        packed_input_int_between_season = torch.nn.utils.rnn.pack_padded_sequence(packed_input_int_between_season,
                                                                                  input_len_int_syn_between_season.tolist(),
                                                                                  batch_first=True,
                                                                                  enforce_sorted=False
                                                                                 )
        
        return packed_input_int, packed_input_int_between_season, input_len_int_syn,\
                (gather_ind, gather_ind_for_multipred)

    def ext_lstm_dt(self, final_week, input_len_int_syn, final_dnn, step) :
        
        final_week = self.embedding(final_week) # (bs, config.p, dim)
        
        bs = final_week.size(0)
        T = input_len_int_syn[-1].to(torch.long) # valid_len + step
        
        input_ext_dec = final_week.unsqueeze(1).expand(bs, 30, self.config.p, self.hidden_size)
        input_ext_dec = input_ext_dec.reshape(-1, self.config.p, self.hidden_size) # (bs*mass, pred_len, self.hidden_size)
        week_emb = input_ext_dec[:,step,:]
        input_ext_dec = input_ext_dec[:,:step,:] # (bs*mass, step, self.hidden_size)
        
        if T>50:
            input_len__50_ext = torch.tensor([50]*(self.batch_sz*30)) + step
            final_week = torch.cat([final_week for i in range(T//self.config.p+1)], 
                                       axis=1)[:,-50:,:]
            final_week = final_week.squeeze()
            final_week = final_week.unsqueeze(1).expand(bs, 30, final_week.size(-2), self.hidden_size).contiguous()
            final_week = final_week.view(-1,final_week.size(-2), self.hidden_size) # (bs*mass, len, self.hidden_size)
            if step>0:
                final_week = torch.cat((final_week, input_ext_dec), dim=1) # (bs*mass, len+step, self.hidden_size)
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
            
        input_len_ext_syn_between_season = input_len_int_syn / self.sw # (bs*mass)
        input_len_ext_syn_between_season_trunc = input_len_ext_syn_between_season.to(torch.int)
        ceil_mask = (input_len_ext_syn_between_season - input_len_ext_syn_between_season_trunc) < 1e-4
        input_len_ext_syn_between_season = torch.ceil(input_len_ext_syn_between_season).to(torch.int)
        input_len_ext_syn_between_season[ceil_mask] = input_len_ext_syn_between_season_trunc[ceil_mask]
        
        max_validlen_between_season = torch.max(input_len_ext_syn_between_season).to(torch.long)
        if step>0:
            packed_input_ext_between_season = input_ext_dec[:,[-1],:].expand(-1, max_validlen_between_season,-1)  
        else:
            packed_input_ext_between_season = final_week[:,[-1],:].expand(-1, max_validlen_between_season,-1)  
        # packed_input_ext_between_season: (bs*mass, max_validlen, self.hidden_size)
        packed_input_ext_between_season = torch.nn.utils.rnn.pack_padded_sequence(packed_input_ext_between_season,
                                                                                  input_len_ext_syn_between_season.tolist(),
                                                                                  batch_first=True,
                                                                                  enforce_sorted=False
                                                                                 )

        return packed_input_ext, packed_input_ext_between_season, week_emb


    def syn_lstm_dt(self, data_syn, week_emb,
                    packed_within_season, packed_between_season,
                    input_lens, severity_hat, gather_inds, step):
        """
        @packed_within_season: (packed_input_int, packed_input_ext)
            |packed_input_ext|: (bs*mass, len, dim)
        @packed_between_season: (packed_input_int_bet_ssn, packed_input_ext_bet_ssn)
            |packed_input_int_bet_ssn|: (bs*mass, [variable/self.sw], dim)
        @input_lens: (input_len,input_len_int_syn)
        @severity_hat: (bs*mass, step, dim)
        """
        
        packed_input_int, packed_input_ext = packed_within_season
        packed_input_int_bet_ssn, packed_input_ext_bet_ssn = packed_between_season
        input_len, input_len_int_syn = input_lens
        gather_ind, gather_ind_for_multipred = gather_inds
        if step>0:
            gather_ind_all = torch.cat((gather_ind, gather_ind_for_multipred), dim=-1) 
        else:
            gather_ind_all = gather_ind
        # gather_ind_all: (bs*mass, 50+step)
        
        T = input_len_int_syn[-1].to(torch.long)
        bs = data_syn.size(0)
        
        data_syn = data_syn.view(bs, data_syn.size(1), self.config.c, -1)
        data_syn = data_syn.transpose(1,2).contiguous().view(data_syn.size(0)*self.config.c,
                                                             data_syn.size(1), -1)
        # data_syn: (bs*#mass, len, dim)
        bs_mass = data_syn.size(0)

        # get severity_answer: target data
        end = -self.config.p+step+1
        if end:
            severity_answer = data_syn[:,-self.config.p+step:end,:].squeeze() #(bs*#mass, 1, dim)
        else: 
            severity_answer = data_syn[:,-self.config.p+step:,:].squeeze() #(bs*#mass, 1, dim)
            
        if step>0:
            data_syn[torch.arange(bs_mass).unsqueeze(-1), gather_ind_for_multipred] = severity_hat
        
        
        # get severity_between_season ######################################################
        uniq_val, inv_ind = input_len_int_syn.unique(return_inverse=True)
        uniq_val = uniq_val.to(torch.long)
        uniq_val_dict = {_: torch.flip(torch.arange(uniq_val[_]-1, -1, -self.sw), dims=[0]) 
                         for _ in range(uniq_val.size(0))}
        gather_ind_bet_sns = torch.nn.utils.rnn.pad_sequence([uniq_val_dict[l.item()] for l in inv_ind],
                                                     batch_first = True,
                                                     padding_value = -1.).to(torch.long) 
        # gather_ind_bet_sns: (bs*mass, self.max_len // self.sw)

        severity_between_season = data_syn[:,:T,:] # (bs*mass, max_validlen, feats)
        severity_between_season = severity_between_season[torch.arange(bs_mass).unsqueeze(-1),
                                                          gather_ind_bet_sns]
        # severity_between_season: (bs*mass, self.validlen // self.sw, feats)
        input_len_int_syn_between_season = input_len_int_syn / self.sw # (bs*mass)
        input_len_int_syn_between_season_trunc = input_len_int_syn_between_season.to(torch.int)
        ceil_mask = (input_len_int_syn_between_season - input_len_int_syn_between_season_trunc) < 1e-4
        input_len_int_syn_between_season = torch.ceil(input_len_int_syn_between_season).to(torch.int)
        input_len_int_syn_between_season[ceil_mask] = input_len_int_syn_between_season_trunc[ceil_mask]
        ####################################################################################
        
        if T>50:
            severity = data_syn[torch.arange(bs_mass).unsqueeze(-1),
                                gather_ind_all]
            severity_shape = severity.size()
        else : 
            severity = data_syn[:,:T,:]
            severity_shape = severity.size()        
        # severity: (bs*#mass, len, dim)

        # pack input
        elapsed, elapsed_len = torch.nn.utils.rnn.pad_packed_sequence(packed_input_int, batch_first=True) 
        weekday, weekday_len = torch.nn.utils.rnn.pad_packed_sequence(packed_input_ext, batch_first=True) 
        
        elapsed_bet_ssn, elapsed_len_bet_ssn = torch.nn.utils.rnn.pad_packed_sequence(packed_input_int_bet_ssn, 
                                                                                      batch_first=True) 
        weekday_bet_ssn, weekday_len_bet_ssn = torch.nn.utils.rnn.pad_packed_sequence(packed_input_ext_bet_ssn, 
                                                                                      batch_first=True) 
        
        packed_input_syn = torch.cat([severity, elapsed, weekday], axis=-1) # data_cat: (bs*mass, T, 41)
        packed_input_syn_bet_ssn = torch.cat([severity_between_season, elapsed_bet_ssn, weekday_bet_ssn], axis=-1)
        # data_cat_bet_ssn: (bs*mass, T//self.se, 41)
        
        packed_input_syn = self.data_emb_linear(packed_input_syn)
        packed_input_syn_bet_ssn = self.data_emb_linear(packed_input_syn_bet_ssn)

        
        if T>50:
            input_len__50 = torch.tensor([50]*packed_input_syn.size(0)) + step
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(packed_input_syn, 
                                                                       input_len__50.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)
        else :
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(packed_input_syn, 
                                                                       input_len_int_syn.tolist(),
                                                                       batch_first=True, 
                                                                       enforce_sorted=False)
            
        packed_input_syn_bet_ssn = torch.nn.utils.rnn.pack_padded_sequence(packed_input_syn_bet_ssn, 
                                                                           input_len_int_syn_between_season.tolist(),
                                                                           batch_first=True, 
                                                                           enforce_sorted=False)
        
        return packed_input_syn, packed_input_syn_bet_ssn, severity_answer
    
    def syn_lstm(self, data_syn, final_dnn, week_emb, covid_mask,
                 packed_within_season, packed_between_season,
                 input_lens, severity_hat, gather_inds, step):
        
        packed_input_syn, packed_input_syn_bet_ssn, severity_answer = self.syn_lstm_dt(data_syn, week_emb, 
                                                                                       packed_within_season, 
                                                                                       packed_between_season,
                                                                                       input_lens, severity_hat,
                                                                                       gather_inds, step)
        
        # input packed_input & output packed_output 
        enc_output, enc_h = self.lstm_within_season(packed_input_syn)
        enc_output_bet_ssn, enc_bet_ssn_h = self.lstm_between_season(packed_input_syn_bet_ssn)
        
        # unpack output 
        enc_output, enc_output_len = torch.nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
        enc_output_bet_ssn, enc_output_bet_ssn_len = torch.nn.utils.rnn.pad_packed_sequence(enc_output_bet_ssn, 
                                                                                            batch_first=True)
        # enc_output/enc_output_bet_ssn: (bs * #massinf, len/len_bet_season, hidden_size//2 * #num_directions=2)
        # enc_h/enc_bet_ssn_h: (#num_directions=2 * #layers, bs*mass, hidden_size//2)
        
        ### Merge within_season & between_season ###
        enc_h, enc_bet_ssn_h = enc_h[0], enc_bet_ssn_h[0]
        bs_mass = enc_h.size(1)
        enc_h = enc_h.view(2, -1, bs_mass, self.hidden_size//2)[:,-1,:,:] # (#dir, #layers, ...)
        enc_h = enc_h.view(2, bs_mass, self.hidden_size//2)
        enc_bet_ssn_h = enc_bet_ssn_h.view(2, -1, bs_mass, self.hidden_size//2)[:,-1,:,:] # (#dir, #layers, ...)
        enc_bet_ssn_h = enc_bet_ssn_h.view(2, bs_mass, self.hidden_size//2)
        
        enc_h = enc_h.transpose(0,1).contiguous().view(bs_mass, -1)
        enc_bet_ssn_h = enc_bet_ssn_h.transpose(0,1).contiguous().view(bs_mass, -1)
        # enc_h/enc_bet_ssn_h: (bs*mass, hidden_size//2 * #num_directions=2)
        
        
        enc_output = self.linear_within_season(enc_h)
        enc_output_bet_ssn = self.linear_between_season(enc_bet_ssn_h)
        # enc_h/enc_bet_ssn_h: (bs*mass, hidden_size=ind)
        
        enc_within_between_season = torch.cat((enc_output, enc_output_bet_ssn), dim=-1)
        severity_hat = self.linear_severity(enc_within_between_season)
        # severity_hat: (bs*mass, severity_emb=6)
        covid_mask = covid_mask.view(-1).to(torch.long)
        try:
            severity_loss = self.loss(severity_hat, severity_answer)[covid_mask] 
        except Exception as e:
            print(severity_hat.size(), severity_answer.size(), step)
            print(e)
            sys.exit(1)
            
        severity_loss = torch.sqrt(severity_loss.mean()) # severity_loss: scalar
        
        enc_within_between_season = self.linear_concat(enc_within_between_season)
        # enc_within_between_season: (bs*mass, hidden_size=ind)
        
        
        ### Encode geographical & economical data
        dnnfeat = self.linear_dnnfeatures(final_dnn.to(torch.float32)) #(bs,mass,hidden)
        dnnfeat = dnnfeat.view(-1, self.hidden_size) #(bs*mass,hidden=ind)
        
        week_emb = self.week_emb_linear(week_emb).squeeze() # (bs*#mass, hidden=34->1) 
        week_emb = week_emb.unsqueeze(-1).expand(-1, self.hidden_size) # (bs*#mass, hidden=ind)
        
        y_hat = torch.cat((enc_within_between_season.unsqueeze(-1),
                           dnnfeat.unsqueeze(-1),
                           week_emb.unsqueeze(-1)), dim=-1)  # (bs*#mass, hidden=ind,3)
        y_hat = self.cat_linear(y_hat).squeeze()
        y_hat = self.final_linear(y_hat) # (bs*#mass, hidden=ind)
        
        return y_hat, severity_loss, severity_hat
    
    
    def compute_y_hat(self, y_hat, covid_mask, modeling_output=None):
        # covid_mask: (bs, #massinf=30)

        y_hat = y_hat.view(self.batch_sz, 30, self.config.p, -1) # (bs, #massinf, config.p, hid)
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
        final_dnn = final_dnn.to(self.config.device, self.config.dtype)
        data_int = data_int.to(self.config.device, self.config.dtype)
        data_syn = data_syn.to(self.config.device, self.config.dtype)
        final_week = final_week.to(self.config.device, )
        final_mask = final_mask.to(self.config.device, )
        covid_mask = covid_mask.to(self.config.device, self.config.dtype)
        
        y_train = y_train.to(self.config.device)
        
        self.batch_sz = data_int.size(0) 
        input_len = final_mask.sum(axis=1)

        y_hat, severity_loss = [], []
        severity_hat = None
        severity_hat_list = []
        for step in range(self.config.p):
            # int_lstm ######################### : output_int
            packed_input_int, packed_input_int_between_season,\
            input_len_int_syn, gather_inds = self.int_lstm_dt(data_int, input_len, step) 

            # ext_lstm ######################### : output_ext
            packed_input_ext, packed_input_ext_between_season, week_emb = self.ext_lstm_dt(final_week,
                                                                                           input_len_int_syn,
                                                                                           final_dnn,
                                                                                           step) 

            # syn_lstm ######################### : output_syn
            y_hat_step, severity_loss_step, severity_hat = self.syn_lstm(data_syn, final_dnn, week_emb,covid_mask,
                                                           (packed_input_int, packed_input_ext),
                                                           (packed_input_int_between_season, 
                                                            packed_input_ext_between_season),
                                                           (input_len,input_len_int_syn), 
                                                           severity_hat, gather_inds, step)
            severity_hat_list.append(severity_hat)
            severity_hat = torch.stack(severity_hat_list, dim=1) # (bs*mass, step, dim=6)
            y_hat.append(y_hat_step)
            severity_loss.append(severity_loss_step)
            
        y_hat = torch.stack(y_hat, dim=1) # (bs*mass, pred, ind)
        severity_loss = torch.stack(severity_loss, dim=-1).mean()
            
        # compute y_hat #########################    
        if inspect:
            logits, modeling_output = self.compute_y_hat(y_hat, covid_mask, modeling_output) # return logits, which is y_hat
            # (bs, pred, ind)
            
        else :
            logits = self.compute_y_hat(y_hat, covid_mask,) # return logits = y_hat
        logits = logits.transpose(1,2).contiguous() # (bs, ind, pred)
        
        # compute MSELoss ######################### logits: (bs, config.p, hid) 
        return (self.loss(logits, y_train.to(torch.float32)), logits, modeling_output) \
                if inspect \
                else (self.loss(logits, y_train.to(torch.float32)), logits, None)
    
    