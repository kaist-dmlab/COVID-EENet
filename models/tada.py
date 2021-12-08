import pandas as pd
import numpy as np
import torch
import torch.nn as nn

class TADA(nn.Module):
    def __init__(self, config, mass_inf = 30, num_regions=25, hidden_size=40, rnn_layer=2,):
        super(TADA, self).__init__()
        
        self.config = config
        self.region = config.r
        self.numOfMassInfection = mass_inf
        self.numOfIndustry = 34
        self.hidden_size = hidden_size
        self.lstm_int = nn.LSTM(input_size=self.numOfMassInfection//self.numOfMassInfection,
               hidden_size=hidden_size,
               num_layers=1, batch_first=True)
        
        self.lstm_ext = nn.LSTM(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1, batch_first=True)
        
        self.lstm_syn = nn.LSTM(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1, batch_first=True)
        self.linear_syn = nn.Linear(hidden_size*2 + 6, hidden_size)

        self.lstm_dec = nn.LSTM(input_size=hidden_size,
                       hidden_size=hidden_size,
                       num_layers=1, batch_first=True)
        # for making decoder input
        self.register_parameter('M_d_con_param', nn.Parameter(torch.randn(hidden_size, hidden_size)))
        self.register_parameter('H_h_int_param', nn.Parameter(torch.randn(hidden_size, hidden_size)))
        self.register_parameter('H_h_ext_param', nn.Parameter(torch.randn(hidden_size, hidden_size)))
        self.register_parameter('V_int', nn.Parameter(torch.randn(hidden_size,1)))
        self.register_parameter('V_ext', nn.Parameter(torch.randn(hidden_size,1)))
        
        self.linear_dnnfeatures = nn.Linear(self.config.dnnfeat_dim, hidden_size)
        self.dec_linear = nn.Linear(hidden_size*2, hidden_size)
        self.linear_align = nn.Linear(hidden_size*2,hidden_size)
        self.linear_y_hat = nn.Linear(hidden_size,self.numOfIndustry)
        
        
        self.embedding = nn.Embedding(7, hidden_size)
        self.week_emb_linear = nn.Linear(hidden_size, 1)
        self.loss = torch.nn.MSELoss(reduction="none")

    def int_lstm(self, data_int, input_len):

        data_int_shape = data_int.size() 
        # (batch_size = #dates*#gus, 362=#dates, 30=#mass_inf_cases)
        data_int = data_int.transpose(1,2).contiguous().view(data_int_shape[0]*data_int_shape[2], data_int_shape[1], 1)
        
        input_len_int_syn = input_len.unsqueeze(-1).expand(input_len.size(0),data_int_shape[2]).contiguous().view(-1)


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
        
        # input packed_input & output packed_output 
        packed_output_int, hidden_int = self.lstm_int(packed_input_int.to(torch.float32))
        
        output_int, out_len_int = torch.nn.utils.rnn.pad_packed_sequence(packed_output_int, batch_first=True)
        return output_int, input_len_int_syn

    def ext_lstm(self, final_week, input_len, final_dnn, output_int) :
        # return output_ext
        final_week = self.embedding(final_week) # (bs, config.p, dim)
        if output_int.size(1)==50:
            input_len__50_ext = torch.tensor([50]*self.batch_sz)
            final_week = torch.cat([final_week for i in range(output_int.size(1)//self.config.p+1)], 
                                       axis=1)[:,-output_int.size(1):,:]
            final_week = final_week.squeeze()
            
            packed_input_ext = torch.nn.utils.rnn.pack_padded_sequence(final_week, 
                                                                       input_len__50_ext.tolist(),
                                                                       batch_first=True, enforce_sorted=False)
        else :
            final_week = torch.cat([final_week for i in range(output_int.size(1)//self.config.p+1)], 
                                       axis=1)[:,-output_int.size(1):,:]
            final_week = final_week.squeeze()
            packed_input_ext = torch.nn.utils.rnn.pack_padded_sequence(final_week, 
                                                                       input_len.tolist(),
                                                                       batch_first=True, enforce_sorted=False)
        # input packed_input & output packed_output 
        packed_output_ext, _ = self.lstm_ext(packed_input_ext.to(torch.float32))
        # unpack output 
        output_ext, out_len_ext = torch.nn.utils.rnn.pad_packed_sequence(packed_output_ext, batch_first=True)
        
        output_ext = output_ext.unsqueeze(1).expand(output_ext.size(0), 30, output_ext.size(-2), self.hidden_size).contiguous()
        # output_ext: (bs, mass, pred, hidden)
        
        output_ext = output_ext.view(-1,output_ext.size(-2), self.hidden_size)

        final_week_dec = final_week[:,-self.config.p:,:]
        
        return output_ext, final_week_dec

    def syn_lstm(self, data_syn, input_len, output_int, output_ext, input_len_int_syn):
        # return output_syn
        cut_t = input_len[-1].to(torch.long)
        if output_int.size(1)==50:
            data_syn = data_syn[:,cut_t-50:cut_t,:]
            data_syn_shape = data_syn.size()
        else : 
            data_syn = data_syn[:,:cut_t,:]
            data_syn_shape = data_syn.size()
        
        data_syn = data_syn.view(data_syn_shape[0], data_syn_shape[1], self.config.c, -1)
        data_syn = data_syn.transpose(1,2).contiguous().view(data_syn.size(0)*self.config.c, data_syn_shape[1], -1)
        
        # pack input
        data_syn_cat = torch.cat([data_syn[:,:output_ext.size(1),:], output_int, output_ext], axis=-1)
        data_syn_cat = self.linear_syn(data_syn_cat)
        
        if output_int.size(1)==50:
            input_len__50 = torch.tensor([50]*output_int.size(0))
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(data_syn_cat, 
                                                               input_len__50.tolist(),
                                                               batch_first=True, enforce_sorted=False)
        else :
            packed_input_syn = torch.nn.utils.rnn.pack_padded_sequence(data_syn_cat, 
                                                               input_len_int_syn.tolist(),
                                                               batch_first=True, enforce_sorted=False)
        # input packed_input & output packed_output 
        packed_output_syn, _ = self.lstm_syn(packed_input_syn.to(torch.float32))
        
        # unpack output 
        output_syn, out_len_syn = torch.nn.utils.rnn.pad_packed_sequence(packed_output_syn, batch_first=True)
        return output_syn
    
    def decoder_lstm_attn(self, output_syn, output_int, output_ext, input_len_int_syn):
        
        extended_bs = output_syn.size(0)
        if output_int.size(1)==50:
            d = output_syn[torch.arange(extended_bs),-1:,:].contiguous()
            c = output_syn[torch.arange(extended_bs),-1:,:].contiguous() 
        else:
            d = output_syn[torch.arange(extended_bs),input_len_int_syn.to(torch.long)-1,:].unsqueeze(1).contiguous() 
            c = output_syn[torch.arange(extended_bs),input_len_int_syn.to(torch.long)-1,:].unsqueeze(1).contiguous()
            
        d_list = []
        output_int = output_int.contiguous()
        output_ext = output_ext.contiguous()
        
        # make attn_mask
        if not output_int.size(1)==50:
            uniqs, cnt = input_len_int_syn.unique(sorted=True,return_counts=True)
            attn_mask = torch.ones(extended_bs, output_int.size(1)).to(self.config.device, torch.long) # (bs,T)
            e = uniqs[0]
            row_s = torch.tensor(0)
            row_e = cnt[0]

            for i in range(1, len(cnt)+1):
                attn_mask[row_s:row_e, :e.to(torch.long)] = torch.tensor(0).to(self.config.device, torch.long)
                if i == len(cnt): continue
                e = uniqs[i]
                row_s = row_e.detach().clone()
                row_e += cnt[i]
            attn_mask = attn_mask.to(torch.bool)

        for _ in range(self.config.p):
            # weighted output_int
            e_input_d_int = (torch.mm(d.squeeze(), self.M_d_con_param).unsqueeze(1) +\
                             torch.mm(output_int.view(-1,output_int.size(-1)), 
                                      self.H_h_int_param).contiguous().view(d.size(0),-1,d.size(-1)) ) # bs, T, hid
            e_input_d_int = nn.Tanh()(e_input_d_int)
            e_input_d_int = torch.mm(e_input_d_int.view(-1,e_input_d_int.size(-1)).contiguous(), self.V_int) # (bs*T, 1)
            e_input_d_int = torch.exp(e_input_d_int.view(d.size(0),-1).contiguous()) # bs, T 
            if not output_int.size(1)==50: # mask
                e_input_d_int = e_input_d_int.masked_fill(attn_mask, 0.)

            alpha_input_d_int = e_input_d_int / (e_input_d_int.sum(-1).unsqueeze(-1)) # (bs, T)
            attn_dec_int = torch.bmm(alpha_input_d_int.unsqueeze(1),
                                     output_int).squeeze() # (bs,1, T)x(bs*30,T,hid)->(bs,hid)
            
            # weighted output_ext
            e_input_d_ext = (torch.mm(d.squeeze(), self.M_d_con_param).unsqueeze(1) +\
                             torch.mm(output_ext.view(-1,output_ext.size(-1)), 
                                      self.H_h_ext_param).contiguous().view(d.size(0),-1,d.size(-1))) # bs, T, hid
            e_input_d_ext = nn.Tanh()(e_input_d_ext)
            e_input_d_ext = torch.mm(e_input_d_ext.view(-1,e_input_d_ext.size(-1)).contiguous(), self.V_ext) # (bs*T, 1)
            e_input_d_ext = torch.exp(e_input_d_ext.view(d.size(0),-1)).contiguous() # bs, T
            if not output_int.size(1)==50: # mask
                e_input_d_ext = e_input_d_ext.masked_fill(attn_mask, 0.)
            alpha_input_d_ext = e_input_d_ext / (e_input_d_ext.sum(-1).unsqueeze(-1)) # (bs, T)
            attn_dec_ext = torch.bmm(alpha_input_d_ext.unsqueeze(1),
                                     output_ext).squeeze() # (bs, 1, T)x(bs*30,T,hid)->(bs,hid)
            
            
            x_dec = torch.cat([attn_dec_int, attn_dec_ext], axis=-1)
            x_dec = self.dec_linear(x_dec)
            x_dec = x_dec.unsqueeze(1)
            
            x_dec, (d, c) = self.lstm_dec(x_dec, (d.transpose(0,1), c.transpose(0,1))) 
            d, c = d.transpose(0,1).contiguous(), c.transpose(0,1).contiguous()
            # x_dec: (bs, hid)
            d_list.append(x_dec) # Eventually, 14-len or 28-len list
            
        d_list = torch.cat(d_list, dim=1)
        return d_list

    def align_trend(self, d_list, output_syn, ):
        align_cand = [output_syn[:,i:i+self.config.p,:] for i in range(output_syn.size(1)) if i+self.config.p < output_syn.size(1)]
        align_cand = torch.stack(align_cand, axis=1)
        align_cand = align_cand.view(align_cand.size(0),align_cand.size(1),-1) # (bs, #cands, config.p*hid)
        align = d_list.view(align_cand.size(0), -1,1) # (bs, config.p*hid, 1)
        align_indice = torch.bmm(align_cand,align).max(dim=1)[1] 

        # merge align candidate ######################### (bs, config.p, 2*hid)
        align_cat = torch.cat([align.view(align.size(0),self.config.p, -1),
                               align_cand[torch.arange(align_indice.size(0)),
                                          align_indice.squeeze(),:].view(align_cand.size(0),
                                                                         self.config.p,
                                                                         -1)],dim=-1)
        return align_cat 
    
    def compute_y_hat(self, align_cat, covid_mask,final_dnn, week_emb, modeling_output=None):
        # covid_mask: (bs, #massinf=30)
        logits = self.linear_align(align_cat.view(align_cat.size(0)*self.config.p, -1)) #(bs*config.p, 2*hid->hid)
        
        dnnfeat = self.linear_dnnfeatures(final_dnn.to(torch.float32)) #(bs, mass, hidden)
        dnnfeat = dnnfeat.view(self.batch_sz*30, -1).unsqueeze(1).expand(-1,self.config.p,-1).contiguous() 
        dnnfeat = dnnfeat.view(self.batch_sz*30*self.config.p, -1)# dnnfeat: (bs*mass*pred, hid)
        
        week_emb = week_emb.unsqueeze(1).expand(-1,30,-1,-1).contiguous()
        week_emb = week_emb.view(self.batch_sz*30*self.config.p, -1)
        
        logits = logits + dnnfeat + week_emb
        logits = nn.Tanh()(logits) # logits: (bs*mass*pred, hid)
        logits = self.linear_y_hat(logits) # 
        
        logits = logits.view(align_cat.size(0),self.config.p,-1) #(bs(=bs * #massinf), config.p, hid) 
        logits = logits.view(self.batch_sz,30,self.config.p,-1) # (bs, #massinf, config.p, hid)
        logits = logits * covid_mask.unsqueeze(-1).unsqueeze(-1)
        if modeling_output :
            modeling_output["logits_bef_mean"] = logits.cpu()
            
        logits = logits.sum(dim=1) # (bs, config.p, hid)
        logits = logits / covid_mask.sum(dim=1).unsqueeze(-1).unsqueeze(-1)
        if modeling_output :
            modeling_output["logits_yhat"] = logits.cpu()

        return logits if modeling_output is None else (logits, modeling_output)
    
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
        output_int, input_len_int_syn = self.int_lstm(data_int, input_len, ) # return output_int
        
        # ext_lstm ######################### : output_ext
        output_ext, final_week_dec = self.ext_lstm(final_week, input_len, final_dnn, output_int) # return output_ext
        
        # syn_lstm ######################### : output_syn
        output_syn = self.syn_lstm(data_syn, input_len, output_int, output_ext, input_len_int_syn) # return output_syn
        
        # decoder with lstm_attn ######################### 
        d_list = self.decoder_lstm_attn(output_syn, output_int, output_ext, input_len_int_syn) # return d_list # (bs, config.p, hid)

        # sample align candidate ######################### 
        align_cat = self.align_trend(d_list, output_syn) # return align_cat: (bs, config.p, 2*hid)
        
        if verbose:
            print(align_cat[...,:align_cat.size(-1)//2], align_cat[...,align_cat.size(-1)//2:])
            print(align_cat[0,0,:])
            
        week_emb = self.week_emb_linear(final_week_dec).squeeze() # (bs, pred)
        week_emb = week_emb.unsqueeze(-1).expand(-1,-1, self.hidden_size)  # (bs, pred, ind)
        
        # compute y_hat #########################
        if inspect:
            modeling_output["output_internal"] = output_int.cpu()
            modeling_output["output_external"] = output_ext.cpu()
            modeling_output["output_syn"] = output_syn.cpu()
            modeling_output["decoder_outputs"] = d_list.cpu()
            modeling_output["align_concated"] = align_cat.cpu()
            
            logits, modeling_output = self.compute_y_hat(align_cat, covid_mask, final_dnn, week_emb,modeling_output) # return logits, which is y_hat
            # (bs, pred, ind)
            
        else :
            logits = self.compute_y_hat(align_cat, covid_mask, final_dnn, week_emb) # return logits = y_hat
            
        
        logits = logits.transpose(1,2).contiguous() # (bs, ind, pred)
        # compute MSELoss ######################### 
        return (self.loss(logits, y_train.to(torch.float32)), logits, modeling_output) \
                if inspect \
                else (self.loss(logits, y_train.to(torch.float32)), logits, None)
    
    