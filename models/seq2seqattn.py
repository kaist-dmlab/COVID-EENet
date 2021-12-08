import pandas as pd
import numpy as np
import torch
import torch.nn as nn
    

class DecoderAttention(nn.Module):
    def __init__(self, config, hidden_size=40):
        super(DecoderAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.directions = 2
        self.linear = nn.Linear(self.hidden_size,
                                self.hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, enc_out, dec_out, attn_mask):
        """
        @enc_out: (bs*massinf, src_len, hidden_size*2)
        @dec_out: (bs*massinf, pred_len, hidden_size*2)
        @attn_mask: (bs*massinf, src_len)
        
        return context_vector: # (bs*massinf, pred_len, hidden_size*2)
        """
        query = self.linear(dec_out)
        # |query| = (bs*massinf, pred_len, hidden_size*2)
        
        weight = torch.bmm(query, enc_out.transpose(1, 2))
        # |weight| = (bs*massinf, pred_len, src_len)
        
        src_len, pred_len = enc_out.size(1), dec_out.size(1)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, pred_len, 1) # (bs*massinf,pred_len,src_len)
        
        weight.masked_fill_(attn_mask, -float('inf'))
        weight = self.softmax(weight) # (bs*massinf, pred_len, src_len)

        context_vector = torch.bmm(weight, enc_out)
        # |context_vector| = (bs*massinf, pred_len, hidden_size*2)

        return context_vector
        
        

class Seq2SeqATTN(nn.Module):
    def __init__(self, config, mass_inf = 30, num_regions=25, hidden_size=40, rnn_layer=1, dropout_p=0.1):
        super(Seq2SeqATTN, self).__init__()
        
        self.config = config
        self.region = config.r
        self.numOfMassInfection = mass_inf
        self.numOfIndustry = hidden_size
        self.severity_emb = 6
        self.hidden_size = hidden_size
        
        self.lstm_syn = nn.LSTM(input_size=hidden_size+self.severity_emb+1,#41
                                hidden_size=hidden_size//2,
                                num_layers=1,
                                bidirectional=True,
                                batch_first=True)
        self.linear_enc2dec_severity = nn.ModuleList([nn.Linear(in_features=self.hidden_size,
                                                       out_features=self.hidden_size) for i in range(2)])        

        self.severity_rnn = nn.LSTM(
            input_size= self.severity_emb,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )
        input_size = 1 + self.severity_emb + self.hidden_size
        transform_rnn_inp_size = input_size - self.severity_emb+ self.hidden_size
        
        self.transform_rnn = nn.LSTM(
            input_size= transform_rnn_inp_size,
            hidden_size= self.hidden_size//2,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
        self.attn = DecoderAttention(config,hidden_size=hidden_size)
        self.concat_linear = nn.Linear(self.hidden_size*2, self.hidden_size) 
        self.tanh = nn.Tanh()
        # MSE loss for severity
        self.severity_linear = nn.Linear(in_features= self.hidden_size, out_features= self.severity_emb)
        
        self.linear_dnnfeatures = nn.Linear(self.config.dnnfeat_dim, self.hidden_size)
        self.week_emb_linear = nn.Linear(self.hidden_size, 1)
        self.linear_y_hat = nn.Linear(hidden_size,34)
        self.last_linear = nn.Linear(config.p,config.p)
        
        self.embedding = nn.Embedding(7, self.hidden_size)
        self.loss = torch.nn.MSELoss(reduction="none")
        
        
    def generate_attnmask(self, x, input_len):
        """
        @x: (bs * #massinf,  self.max_len or var_len, cat_feats) to get dtype
        @input_len: (bs * #massinf)
        
        return mask: (bs * #massinf, max_len of input_len)
        """
        
        mask = []
        
        input_len = input_len.to(torch.long).tolist()
        max_length = max(input_len)
        for l in input_len:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples, 
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, (max_length - l))
                                    ], dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples, 
                # set every value in mask to be 0.
                mask += [x.new_ones(1, l).zero_()] # (l,)

        mask = torch.cat(mask, dim=0).bool()

        return mask.to(self.config.device) # (bs * #massinf, max_len of input_len)
    
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
        input_ext_dec = input_ext_dec.reshape(-1, self.config.p, self.hidden_size) # (bs*mass, pred_len, self.hidden_size)
        
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


    def syn_lstm(self, data_syn, final_dnn, input_len, packed_input_int, packed_input_ext, input_len_int_syn, ):
        """
        @
        """
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
        # data_cat: (bs*mass, T, 41)

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
        enc_output, enc_h = self.lstm_syn(packed_input_syn.to(torch.float32))
        
        # unpack output 
        enc_output, enc_output_len = torch.nn.utils.rnn.pad_packed_sequence(enc_output, batch_first=True)
        # enc_output: (bs * #massinf, len, hidden_size//2 * #num_directions=2)
        
        # add dnn features
        dnnfeat = self.linear_dnnfeatures(final_dnn.to(torch.float32)) #(bs, hidden)
        dnnfeat = dnnfeat.unsqueeze(1).unsqueeze(1) #(bs*mass, 1, hidden)
        # dnnfeat:(bs,1,1,hidden)
        enc_output = enc_output.view(bs*self.numOfMassInfection, -1, self.hidden_size)
        return enc_output, enc_output_len, enc_h, severity_dec
    
    def decoder_lstm_attn(self, enc_output, enc_output_len, enc_h,covid_mask, final_dnn, decoder_inp):
        """
        @enc_out: (bs * #massinf, len, hidden_size * #num_directions=2)
        @enc_output_len: (bs * #massinf)
        @enc_h = (h,c): (#layers=1 * #num_dir=2, bs*#massinf, hidden_size)
        
        @decoder_inp: Tuple(internal_dt: (bs*#mass, pred_len, 1),
                            external_dt: (bs*#mass, pred_len, 34),
                            sevirity_dt: (bs*#mass, pred_len+1, 6),)
        """
        
        src_padding_mask = self.generate_attnmask(enc_output, enc_output_len) # (bs * #massinf, src_len)
        
        d_list = []
        severity_dec = decoder_inp[2]
        severity_src = severity_dec[:,:-1,:]
        severity_tgt = severity_dec[:,1:,:] # (bs*#massinf, pred_len, severity_emb)
        
        bs_massinf = enc_h[0].size(1)
        hidden_severity_rnn = [h_c.transpose(0,1).contiguous().view(bs_massinf,-1,self.hidden_size).transpose(0,1).contiguous()\
                               for h_c in enc_h] # (#layers, bs*massinf, hidden*2)
        hidden_severity_rnn = [h_c.view(-1, self.hidden_size) for h_c in hidden_severity_rnn]# (#layers* bs*massinf, hidden*2)
        hidden_severity_rnn = [self.linear_enc2dec_severity[i](hidden_severity_rnn[i]) for i in range(len(hidden_severity_rnn))]
        hidden_severity_rnn = [h_c.view(-1,bs_massinf,self.hidden_size) for h_c in hidden_severity_rnn]
        # hidden_severity_rnn: (#layers, bs*massinf, hidden)
        
        if self.training: # Teacher forcing
            hidden_severity, sev_h = self.severity_rnn(severity_src, hidden_severity_rnn)
            # epidemiological_severity_dec: (bs*#massinf, #pred, dim=6)
            # sev_h[0]: (#layers(=1) * #directions(=1), bs*#massinf, hidden)

        else : # AR
            hidden_severity = []

            for t in range(self.config.p):
                sev_inp_t = severity_src[:,[t],:] # (bs*#massinf, 1, dim=6) 
                sev_inp_t_out, hidden_severity_rnn = self.severity_rnn(sev_inp_t, hidden_severity_rnn)
                # sev_inp_t_out: (bs*#massinf, 1, hidden_size) 
                hidden_severity.append(sev_inp_t_out)

            assert len(hidden_severity) == self.config.p
            hidden_severity = torch.cat(hidden_severity, dim=1) # (bs*#massinf, pred_len, hidden_size) 
                
        # computes mseloss: hidden_severity ~ epidemiological_severity_tgt
        hidden_severity = hidden_severity.contiguous()
        hidden_severity_tgt_hat = self.severity_linear(hidden_severity.view(-1, self.hidden_size))
        hidden_severity_tgt_hat = hidden_severity_tgt_hat.view(-1, self.config.p, self.severity_emb)
        covid_mask = covid_mask.view(-1).to(torch.long) # (bs*#massinf,) active mass inf if 1
        severity_rmseloss = self.loss(hidden_severity_tgt_hat, severity_tgt)[covid_mask]
        # severity_rmseloss: (none: bs * #active_massinf, pred_len, hidden_size=6)
        severity_rmseloss = torch.sqrt(severity_rmseloss.mean()) 
        
        covid_elapsed_dec, weekdays_dec = decoder_inp[0], decoder_inp[1]
        transform_inp = torch.cat([weekdays_dec, covid_elapsed_dec, hidden_severity], dim = -1)
        transform_inp = transform_inp.to(self.config.dtype)
        
        dec_out, dec_h = self.transform_rnn(transform_inp, enc_h)
        # (bs*#massinf, config.p, hidden_size * #directions=2)

        # use attention
        # enc_output: (bs*#massinf, src_len, hidden_size * #directions=2)
        context_vector = self.attn(enc_output, dec_out, src_padding_mask) # (bs*massinf, pred_len, hidden_size)
        dec_out = self.tanh(self.concat_linear(torch.cat([dec_out,context_vector], dim=-1)))
        # dec_out: (bs*massinf, pred_len, hidden_size)

        dnnfeat = self.linear_dnnfeatures(final_dnn.to(torch.float32)) #(bs,mass,hidden)
        dnnfeat = dnnfeat.view(-1, self.hidden_size).unsqueeze(1) #(bs*mass,1,hidden)
        
        week_emb = self.week_emb_linear(weekdays_dec).squeeze() # (bs*#mass, pred_len, 1) 
        week_emb = week_emb.unsqueeze(-1).expand(-1,-1, self.numOfIndustry) # (bs*#mass, pred_len, 34=ind) 
        
        dec_out = dec_out + dnnfeat + week_emb
        # dec_out: (bs*massinf, pred_len, hidden_size)
        dec_out = dec_out.contiguous().view(-1, self.hidden_size)
        dec_out = self.tanh(self.linear_y_hat(dec_out))
        y_hat = dec_out.view(-1, self.config.p, 34) # (bs*#massinf, pred_len, hidden_size=34)
        
        return y_hat, severity_rmseloss
    
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
        enc_output, enc_output_len, enc_h, severity_dec = self.syn_lstm(data_syn, final_dnn, input_len,
                                                                        packed_input_int, packed_input_ext, 
                                                                        input_len_int_syn, )
        # return output_syn
        
        # decoder with lstm_attn ######################### 
        y_hat, severity_rmseloss = self.decoder_lstm_attn(enc_output, enc_output_len, enc_h, covid_mask, final_dnn,
                                                            decoder_inp=(input_int_dec, input_ext_dec, severity_dec)) 
        # return y_hat (bs*#massinf, pred_len, hidden_size=34)
    
    
        if inspect:
            logits, modeling_output = self.compute_y_hat(y_hat, covid_mask, modeling_output) # return logits, which is y_hat
            
        else :
            logits = self.compute_y_hat(y_hat, covid_mask,) # return logits = y_hat
            
        logits = logits.transpose(1,2).contiguous() # (bs, ind, pred)
        logits = self.last_linear(logits)
        # compute MSELoss ######################### logits: (bs, config.p, hid) 
        return (self.loss(logits, y_train.to(torch.float32))+0.1*severity_rmseloss, logits, modeling_output) \
                if inspect \
                else (self.loss(logits, y_train.to(torch.float32)), logits, None)
    
    