import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from model_utils import make_input_for_epidemic_encoder

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = config.h
        self.embedding_dim = config.e
        self.split_size = self.embedding_dim // self.num_heads
        self.numOfIndustry = config.n

        self.linears_Q = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) 
                                      for i in range(self.num_heads)])
        self.linears_K = nn.ModuleList([nn.Linear(self.embedding_dim, self.embedding_dim, bias=False) 
                                      for i in range(self.num_heads)])
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, district_buz_emb, district_indice):
        """
        @param district_buz_emb: (#districts * #buz, emb_dim)
        @param district_indice: (#districts)
        
        return attn: (#districts, #ind, #ind) * #heads
                     sum of attn (i, :, j) = 1
        1. indexing emb_layer by district_indice 
        3. make attn tensor: (#districts, #ind, #ind) per head
        """ 
        # 1. indexing emb_layer by district_indice 
        e_weight = torch.split(district_buz_emb, self.numOfIndustry) # (34, emb_dim) * num_districts
        e_weight = torch.cat([e_weight[index] for index in district_indice],
                             dim=0) # ((#regions or #inf_regions)*34, emb_dim)
        
        # 2. make attn tensor: (#districts, #ind, #ind) per head
        QK_heads = []
        for i, (linear_Q, linear_K) in enumerate(zip(self.linears_Q, self.linears_K)):
            Q = linear_Q(e_weight) # (#r or #i_r * 34, emb_dim)
            Q = Q.view(district_indice.size(0), self.numOfIndustry, self.embedding_dim)
            K = linear_K(e_weight) # (#r or #i_r * 34, emb_dim)
            K = K.view(district_indice.size(0), self.numOfIndustry, self.embedding_dim)
            
            QK = torch.bmm(Q,K.transpose(1,2))/(torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))) # (#r or #i_r, #ind, #ind)
            QK = self.softmax(QK)
            QK_heads.append(QK) 
                                        
        return QK_heads # (#r or #i_r, #ind, #ind) * heads
        

class EconomicViewSubEncoder(nn.Module):
    def __init__(self, config):
        super(EconomicViewSubEncoder, self).__init__()
        
        self.region = config.r
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        self.num_consumer = 27
        self.num_heads = config.h
        self.embedding_dim = config.e
        self.activation = config.activation
        
        self.softmax = nn.Softmax(dim=-1)
    
        ## 2. BR
        self.district_buz_emb = nn.Embedding(num_embeddings = self.region*self.numOfIndustry, 
                                             embedding_dim = self.embedding_dim)
        self.district_buz_emb_layernorm = nn.LayerNorm(normalized_shape= [self.embedding_dim], eps = 1e-16, )
        self.multihead_attn_trg = MultiHeadAttention(config)
        self.multihead_attn_infected = MultiHeadAttention(config)
        
        ## 3. BS
        self.cosine = nn.CosineSimilarity(dim=-1, eps=1e-15)
        
        ## 4. CS
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.kld = nn.KLDivLoss(reduction='none', log_target=True)
        
        ## 5. OS
        self.linear_os = nn.Linear(in_features=self.embedding_dim, 
                                   out_features=self.embedding_dim)
        self.BS_layernorm = nn.LayerNorm(normalized_shape=[self.numOfMassInfection, self.numOfIndustry])
        self.CS_layernorm = nn.LayerNorm(normalized_shape=[self.numOfMassInfection, self.numOfIndustry])
        self.OS_layernorm = nn.LayerNorm(normalized_shape=[self.numOfMassInfection, self.numOfIndustry])
        
        
    def forward(self, 
                business_structure_target, \
                business_structure_infected, \
                customer_structure_target, \
                customer_structure_infected, \
                index_target_idx, \
                index_infected_idx, 
                covid_outbreak_business, ): # output (bs, c, #industry)
        """
        @param business_structure_target: (bs, #massinf, #ind, #feat=4)
        @param business_structure_infected: (bs, #massinf, #ind, #feat)
        @param customer_structure_target: (bs, #massinf, #ind, #cust, #feat)
        @param customer_structure_infected: (bs, #massinf, #ind, #cust, #feat)
        @param index_target_idx: (bs, #massinf): its index, same index across #massinf
        @param index_infected_idx: (bs, #massinf): same index across samples in batch
        """
        
        business_structure_target = (business_structure_target).mean(dim=-1) # (bs, #massinf, #ind)
        business_structure_infected = (business_structure_infected).mean(dim=-1)
        customer_structure_target = (customer_structure_target).mean(dim=-1) # (bs, #massinf, #ind, #cust)
        customer_structure_infected = (customer_structure_infected).mean(dim=-1)
    
        bs = business_structure_target.size(0)
        rep = bs//self.region
        business_structure_target = business_structure_target[:self.region, 0, :] # (#regions, #ind)
        business_structure_infected = business_structure_infected[0,:,:] # (#infected_regions, #ind) 
        customer_structure_target = customer_structure_target[:self.region, 0, :,:] # (#regions, #ind, #cust)
        customer_structure_infected = customer_structure_infected[0,:,:,:] # (#infected_regions, #ind, #cust)
        
        # 2. Buz-struct Representation (BR): (#regions, #ind, num_heads)
        attn_trg = self.multihead_attn_trg(self.district_buz_emb_layernorm(self.district_buz_emb.weight), 
                                           index_target_idx.squeeze()[:25,0].to(torch.long)) # (#regions, #ind, #ind) * #heads
        BR_trg = [torch.bmm(business_structure_target.unsqueeze(1), 
                            attn).squeeze() for attn in attn_trg] # (#regions, 1, #ind) * (#regions, #ind, #ind) -> (#regions, #ind) * #heads
        BR_trg = torch.stack(BR_trg, dim=-1) # (#regions, #ind, num_heads)
        
        attn_inf = self.multihead_attn_infected(self.district_buz_emb_layernorm(self.district_buz_emb.weight), 
                                                index_infected_idx.squeeze()[0].to(torch.long)) # (#infected_regions, #ind, #ind) * #heads
        BR_inf = [torch.bmm(business_structure_infected.unsqueeze(1), 
                            attn).squeeze() for attn in attn_inf] # (#i_regions, 1, #ind) * (#i_regions, #ind, #ind) -> (#i_regions, #ind) * #heads
        BR_inf = torch.stack(BR_inf, dim=-1) # (#infected_regions, #ind, num_heads)
        
        # 3. Buz-struct Similarity (BS): (#regions x #infected_regions) pairs' (#ind) tensor
        BR_trg = BR_trg.unsqueeze(1).repeat(1, BR_inf.size(0), 1, 1)
        BR_trg = BR_trg.reshape(-1, self.num_heads) # (#r x #i_r x #ind, #heads)
        BR_inf = BR_inf.repeat(self.region,1,1).contiguous() # (#i_r x #r,#ind, #heads)
        BR_inf = BR_inf.view(-1, self.num_heads) # (#i_r x #r x #ind, #heads)
        
        BS = self.cosine(BR_trg,BR_inf).view(self.region, -1, self.numOfIndustry) # (#r, #i_r, #ind)
        
        # 4. Consumer-struct Similarity (CS)
        customer_structure_target = customer_structure_target.unsqueeze(1).repeat(1,
                                                                                  customer_structure_infected.size(0),
                                                                                  1,
                                                                                  1)
        customer_structure_target = customer_structure_target.reshape(-1, self.num_consumer) #(#r x #i_r x #ind, #cust)
        customer_structure_target = self.log_softmax(customer_structure_target)
        customer_structure_infected = customer_structure_infected.repeat(self.region, 1, 1)
        customer_structure_infected = customer_structure_infected.view(-1, self.num_consumer) #(#i_r x #r x #ind, #cust)
        customer_structure_infected = self.log_softmax(customer_structure_infected)
        
        M = (customer_structure_target+customer_structure_infected)/2
        M = self.log_softmax(M)
        
        CS = (self.kld(customer_structure_target, M).mean(-1) + self.kld(customer_structure_infected, M).mean(-1))
        CS = (CS/2).view(self.region, -1, self.numOfIndustry) # (#r, #i_r, #ind)
        CS = -CS # to make bigger = more similar (#r, #i_r, #ind)
        
        # 5. Outbreak-buz Similarity (OS)
        covid_outbreak_business = (covid_outbreak_business.squeeze()[0]).to(torch.long) # (#massinf)
        inf_districts = (index_infected_idx.squeeze()[0]).to(torch.long)
        inf_emb_indice = inf_districts*self.numOfIndustry + covid_outbreak_business # (#massinf)
        covid_outbreak_business_emb = self.district_buz_emb(inf_emb_indice) # (#massinf, emb_dim)
        covid_outbreak_business_emb = self.linear_os(covid_outbreak_business_emb) # (#massinf, emb_dim)
        
        all_d_b_emb = self.district_buz_emb.weight #(#regions * #inds, emb_dim)
        OS = torch.mm(covid_outbreak_business_emb, all_d_b_emb.T) # (#massinf, #regions * #inds)
        OS = OS.view(-1, self.region, self.numOfIndustry)
        OS = OS.transpose(0,1).contiguous() # (#r, #i_r = #massinf, #ind)
        
        BS = self.BS_layernorm(BS)
        CS = self.CS_layernorm(CS)
        OS = self.OS_layernorm(OS)
        
        return BS, CS, OS # (#r, #i_r = #massinf, #ind)x3
       

class GeographicViewSubEncoder(nn.Module):
    def __init__(self, config):
        super(GeographicViewSubEncoder, self).__init__()
        self.ab_GER = config.ablation_GER
        self.numOfMassInfection = config.c
        self.linear = nn.Linear(in_features=2, out_features=1)
        self.geographic_layernorm = nn.LayerNorm(normalized_shape=[self.numOfMassInfection], eps=1e-15,)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, physical_distance, contextual_distance):
        """
        @param physical_distance: (bs, #massinf=30)
        @param contextual_distance: (bs, #massinf=30)
        
        return GER: (bs, #massinf) distance between target and massinf
        """
        if self.ab_GER:
            GER = torch.ones_like(physical_distance)
        else:
            physical_distance, contextual_distance = physical_distance.squeeze(), contextual_distance.squeeze()
            dist_cat = torch.stack([physical_distance, contextual_distance], axis=-1) # (bs, #massinf, 2)
            dist_cat = dist_cat.view(-1, dist_cat.size(-1)) # (bs * #massinf, 2)

            GER = self.linear(dist_cat).view(-1, self.numOfMassInfection) # (bs, #massinf)
        GER = self.relu(GER)
        return GER


class EpidemicViewSubEncoder(nn.Module):
    def __init__(self, config):
        super(EpidemicViewSubEncoder, self).__init__()
        
        self.region = config.r
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        self.max_len = config.epidemic_encoder_maxlen
        self.pred_len = config.p
        self.embedding_dim = config.e # 20
        
        self.is_decoder_attn = config.is_decoder_attn
        
        self.week_embs = nn.Embedding(num_embeddings=7, embedding_dim=self.embedding_dim)
        self.week_emb_layernorm = nn.LayerNorm(normalized_shape= [self.week_embs.embedding_dim],
                                               eps = 1e-16, )
        self.sev_embdim = 6
        self.elapsedday_embdim = 6
        
        self.encoder = EpidemicViewEncoder(config, input_size=self.week_embs.embedding_dim+ self.sev_embdim+ self.elapsedday_embdim,)
        self.decoder = EpidemicViewDecoder(config, input_size=self.week_embs.embedding_dim+ self.sev_embdim+ self.elapsedday_embdim, 
                                           severity_emb= self.sev_embdim )
        
        self.config = config
        

    
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
    
    @staticmethod        
    def generate_square_subsequent_mask(sz, device):
        """
        Generate a square mask for the sequence. 
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device) 
    
    def forward(self, epidemiological_severity, covid_elapsed_day, weekdays, mask, covid_mask, covid_start):        
        """
        @param epidemiological_severity: (bs, valid_len+pred_len, #massinf, n_feats=6)
        @param covid_elapsed_day: (bs, valid_len+pred_len, #massinf, dim=6)
        @param weekdays: (bs, #massinf, pred_len, 1)
        @param mask: (bs, #massinf, pred_len, 1); mask[:,0,:,0].sum(1) -> valid_len
        @param covid_mask: (bs, #massinf)
        @param covid_start: dict
        
        return EPR: (bs, #massinf, pred_len, )
        """
        valid_len = mask[:,0,:,0].sum(1) # (bs)
        bs = valid_len.size(0)
        weekdays = weekdays.squeeze()[:,0,:].to(torch.long) # (bs, pred_len)
        weekdays = self.week_emb_layernorm(self.week_embs(weekdays)) # (bs, pred_len, emb_dim = 6)
        
        (inp_enc_cat,input_len), \
        weekdays_dec,\
        covid_elapsed_dec,\
        epidemiological_severity_dec = make_input_for_epidemic_encoder(self, bs, valid_len, weekdays,
                                                                   covid_elapsed_day,
                                                                   epidemiological_severity,
                                                                   covid_mask,covid_start,)
        
        
        # generate_mask: attn_mask - (bs * #massinf, max_len of input_len)
        if self.is_decoder_attn or self.config.tm_seq2seq:
            attn_mask = self.generate_attnmask(inp_enc_cat, input_len) 
        
        # encoder
        enc_out, enc_h = self.encoder(inp_enc_cat, input_len,
                                      tm_mask = attn_mask if self.config.tm_seq2seq else None)
        
        # decoder
        if self.is_decoder_attn:
            EPR, dec_h, severity_mseloss = self.decoder(enc_out, enc_h, weekdays_dec, covid_elapsed_dec, epidemiological_severity_dec,
                                                        attn_mask, covid_mask=covid_mask)
        else:
            EPR, dec_h, severity_mseloss = self.decoder(enc_out, enc_h, weekdays_dec, covid_elapsed_dec, epidemiological_severity_dec,
                                                        None, covid_mask=covid_mask)

        # EPR: (bs*#massinf, self.pred, hidden * #directions)
        return EPR, dec_h, severity_mseloss, weekdays_dec
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1,):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1,15,6) batch_first : 
        pe = pe.to(torch.float32)
        self.register_buffer('pe', pe) # pe: (1, T: len, D: embed_sz)

    def forward(self, x):
        """
        @x: (N,T,D)
        """
        x = x.to(torch.float32)
        x = x + self.pe[:,:x.size(1), :]
        return self.dropout(x)
    

class EpidemicViewEncoder(nn.Module):
    def __init__(self, config, input_size,  dropout_p=0.1):
        super(EpidemicViewEncoder, self).__init__()
        
        hidden_size = config.seq2seq_encoder_lstm_cell # 18
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )

        self.tm_seq2seq = config.tm_seq2seq
        if self.tm_seq2seq:
            
            
            self.linear_input2hidden = nn.Linear(input_size, hidden_size)
            # position encoding
            self.max_len = config.epidemic_encoder_maxlen
            self.pos_enc = PositionalEncoding(d_model=hidden_size, max_len = self.max_len)
            
            # TransformerEncoder, TransformerEncoderLayer
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, 
                                                       nhead=2,
                                                       dim_feedforward=hidden_size,
                                                       dropout=dropout_p, 
                                                       batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                             num_layers=1)
        
        self.hidden_size = hidden_size
        
    def forward(self,inp_enc_cat,input_len, tm_mask=None):
        """
        @inp_enc_cat: (bs * #massinf,  self.max_len or var_len, cat_feats)
        @input_len: (bs*#massinf,)
        @tm_mask: (bs*#massinf, self.max_len or var_len)
        
        *you can use lstm through torch.nn.utils.rnn.pack_padded_sequence
        """
        if self.tm_seq2seq:
            inp_enc_cat = self.linear_input2hidden(inp_enc_cat)
            inp_enc_cat = inp_enc_cat * torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32))
            inp_enc_cat = self.pos_enc(inp_enc_cat)
            # tm_mask # (N,T)
            enc_out = self.transformer_encoder(inp_enc_cat, 
                                               mask=None,
                                               src_key_padding_mask=tm_mask)
            # enc_out: (N,T,D)
            return enc_out, None
            
            
        
        packed_input_enc = torch.nn.utils.rnn.pack_padded_sequence(inp_enc_cat, 
                                                                   input_len.tolist(),
                                                                   batch_first=True, 
                                                                   enforce_sorted=False)

        enc_out, enc_h = self.rnn(packed_input_enc)
        enc_out = torch.nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True, padding_value=0.)
        
        # enc_out=(hidden_rep, hidden_rep_len), enc_h=(h, c)
        # hidden_rep: (bs * #massinf, len, hidden_size * #num_directions=2)
        # hidden_rep_len: (bs * #massinf)
        # h, c: (#layers * #num_dir, bs, hidden_size)
        return enc_out, enc_h 
        

class EpidemicViewDecoder(nn.Module):
    def __init__(self, config, input_size, severity_emb,  dropout_p=0.1):
        super(EpidemicViewDecoder, self).__init__()
        
        self.config = config
        self.hidden_size = config.seq2seq_encoder_lstm_cell # 18
        self.emb_size = config.e
        self.pred_len = config.p
        self.severity_emb=severity_emb
        
        self.is_decoder_attn = config.is_decoder_attn
        
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.mseloss = nn.MSELoss(reduction="none")
        self.linear_enc2dec_severity = nn.ModuleList([nn.Linear(in_features=2*self.hidden_size,
                                                       out_features=self.hidden_size) for i in range(2)])        
        self.severity_rnn = nn.LSTM(
            input_size= self.severity_emb,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True,
        )
        self.severity_linear = nn.Linear(in_features= self.hidden_size, out_features= self.severity_emb)
        
        transform_rnn_inp_size = input_size- self.severity_emb+ self.hidden_size
        
        self.transform_rnn = nn.LSTM(
            input_size= transform_rnn_inp_size,
            hidden_size= self.hidden_size,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
        self.linear = nn.Linear(self.hidden_size*2, 10) 
        
        if self.is_decoder_attn:
            self.attn = DecoderAttention(config)
            self.concat_linear = nn.Linear(self.hidden_size*4, self.hidden_size*2) 
            
        self.tm_seq2seq = config.tm_seq2seq
        if self.tm_seq2seq:
            
            # position encoding
            self.severity_pos_enc = PositionalEncoding(d_model=self.hidden_size, max_len = self.pred_len)
            self.trsf_pos_enc = PositionalEncoding(d_model=self.hidden_size*2, max_len = self.pred_len)
            
            # TransformerEncoder, TransformerEncoderLayer
            self.linear_input2hidden = nn.Linear(self.severity_emb, self.hidden_size)
            sev_decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size, 
                                                           nhead=2,
                                                           dim_feedforward=self.hidden_size,
                                                           dropout=dropout_p, 
                                                           batch_first=True)
            self.severity_transformer_decoder = nn.TransformerDecoder(sev_decoder_layer, 
                                                                      num_layers=1)
            self.severity_tgt_mask = EpidemicViewSubEncoder.generate_square_subsequent_mask(self.pred_len,
                                                                                           device = config.device)
            
            self.linear_tgt_hidden2hidden_2 = nn.Linear(transform_rnn_inp_size, self.hidden_size*2)
            self.linear_memory_hidden2hidden_2 = nn.Linear(self.hidden_size, self.hidden_size*2)
            trsf_decoder_layer = nn.TransformerDecoderLayer(d_model=self.hidden_size*2, 
                                                            nhead=2,
                                                            dim_feedforward=self.hidden_size*2,
                                                            dropout=dropout_p, 
                                                            batch_first=True)
            self.trsf_transformer_decoder = nn.TransformerDecoder(trsf_decoder_layer, 
                                                                  num_layers=1)
            
       
        
    def forward(self, enc_out, enc_h, weekdays_dec, covid_elapsed_dec, epidemiological_severity_dec, 
                attn_mask, covid_mask):
        """
        if self.tm_seq2seq == True:
            @enc_out = (bs * #massinf, len, hidden_size)
            @enc_h = None
        else:
            @enc_out = (hidden_rep, hidden_rep_len)
                    # hidden_rep: (bs * #massinf, len, hidden_size * #num_directions=2)
                    # hidden_rep_len: (bs * #massinf)
            @enc_h = (h,c): (#layers=1 * #num_dir=2, bs*#massinf, hidden_size)
        
        @weekdays_dec: (bs*#massinf, 1+#pred, #dim or #feats)
        @covid_elapsed_dec: (bs*#massinf, 1+#pred, #dim or #feats)
        @epidemiological_severity_dec: (bs*#massinf, 1+#pred, #dim or #feats)
        @attn_mask: (bs, #massinf, src_len)
        @covid_mask: (bs, #massinf)
        
        1. severity_rnn: epidemiological_severity_dec with teacher forcing in training or AR in evaluating.
        2. transform_rnn: [weekdays_dec; covid_elapsed_dec; epidemiological_severity_dec]
        
        """
        # 1. severity_rnn
        
        epidemiological_severity_src = epidemiological_severity_dec[:,:-1,:]
        epidemiological_severity_tgt = epidemiological_severity_dec[:,1:,:] # (bs*#massinf, pred_len, severity_emb) 
            
        if self.tm_seq2seq:
            epidemiological_severity_src = self.linear_input2hidden(epidemiological_severity_src) #to hidden_size
            epidemiological_severity_src = epidemiological_severity_src * torch.sqrt(torch.tensor(self.hidden_size,
                                                                                                  dtype=torch.float32))
            epidemiological_severity_src = self.severity_pos_enc(epidemiological_severity_src)

            hidden_severity = self.severity_transformer_decoder(tgt=epidemiological_severity_src, 
                                                                memory= enc_out,
                                                                tgt_mask= self.severity_tgt_mask, 
                                                                memory_mask=None, 
                                                                tgt_key_padding_mask=None, 
                                                                memory_key_padding_mask= attn_mask)
            # hidden_severity: (bs*#massinf, pred_len, hidden_size)
        else:
            # enc_h to dec_init
            bs_massinf = enc_h[0].size(1)
            hidden_severity_rnn = [h_c.transpose(0,1).contiguous().view(bs_massinf,-1,self.hidden_size*2).transpose(0,1).contiguous()\
                                   for h_c in enc_h] # (#layers, bs*massinf, hidden*2)
            hidden_severity_rnn = [h_c.view(-1, self.hidden_size*2) for h_c in hidden_severity_rnn]# (#layers* bs*massinf, hidden*2)
            hidden_severity_rnn = [self.linear_enc2dec_severity[i](hidden_severity_rnn[i]) for i in range(len(hidden_severity_rnn))]
            hidden_severity_rnn = [h_c.view(-1,bs_massinf,self.hidden_size) for h_c in hidden_severity_rnn]
            # hidden_severity_rnn: (#layers, bs*massinf, hidden)

            if self.training: # Teacher forcing
                hidden_severity, sev_h = self.severity_rnn(epidemiological_severity_src, hidden_severity_rnn)
                # epidemiological_severity_dec: (bs*#massinf, #pred, dim=6)
                # sev_h[0]: (#layers(=1) * #directions(=1), bs*#massinf, hidden)

            else : # AR
                hidden_severity = []

                for t in range(self.pred_len):
                    sev_inp_t = epidemiological_severity_src[:,[t],:] # (bs*#massinf, 1, dim=6) 
                    sev_inp_t_out, hidden_severity_rnn = self.severity_rnn(sev_inp_t, hidden_severity_rnn)
                    # sev_inp_t_out: (bs*#massinf, 1, hidden_size) 
                    hidden_severity.append(sev_inp_t_out)

                assert len(hidden_severity) == self.pred_len
                hidden_severity = torch.cat(hidden_severity, dim=1) # (bs*#massinf, pred_len, hidden_size) 
            
        hidden_severity = hidden_severity.contiguous()
        hidden_severity_tgt_hat = self.severity_linear(hidden_severity.view(-1, self.hidden_size))
        hidden_severity_tgt_hat = hidden_severity_tgt_hat.view(-1, self.pred_len, self.severity_emb)
        covid_mask = covid_mask.view(-1).to(torch.long) # (bs*#massinf,) active mass inf if 1
        severity_rmseloss = self.mseloss(hidden_severity_tgt_hat, epidemiological_severity_tgt)[covid_mask]
        # severity_rmseloss: (none: bs * #active_massinf, pred_len, hidden_size=6)
        severity_rmseloss = torch.sqrt(severity_rmseloss.mean()) 
        
        # 2. transform_rnn
        transform_inp = torch.cat([weekdays_dec[:,1:,:], covid_elapsed_dec[:,1:,:], hidden_severity],
                                      dim = -1) # (bs*#massinf, pred_len, hidden_size)
        if self.tm_seq2seq:
            transform_inp = self.linear_tgt_hidden2hidden_2(transform_inp) #to hidden_size*2
            enc_out = self.linear_memory_hidden2hidden_2(enc_out) #to hidden_size*2
            
            transform_inp = transform_inp * torch.sqrt(torch.tensor(self.hidden_size*2, 
                                                                    dtype=torch.float32),)
            transform_inp = self.trsf_pos_enc(transform_inp)

            dec_out = self.trsf_transformer_decoder(tgt=transform_inp, 
                                                    memory= enc_out,
                                                    tgt_mask= None, # self.severity_tgt_mask
                                                    memory_mask=None, 
                                                    tgt_key_padding_mask=None, 
                                                    memory_key_padding_mask= attn_mask)
            # dec_out: (bs*massinf, pred_len, hidden_size*2)
        
        else:
            dec_out, dec_h = self.transform_rnn(transform_inp, enc_h)
            # (bs*#massinf, pred_len, hidden_size * #directions=2)

            # use attention
            if self.is_decoder_attn:
                enc_out = enc_out[0]
                context_vector = self.attn(enc_out, dec_out, attn_mask) # (bs*massinf, pred_len, hidden_size*2)
                dec_out = self.relu(self.concat_linear(torch.cat([dec_out,context_vector], dim=-1)))
                # dec_out: (bs*massinf, pred_len, hidden_size*2)

        dec_out = dec_out.contiguous().view(-1, self.hidden_size*2)
        dec_out = (self.linear(dec_out))
        dec_out = dec_out.view(-1, self.pred_len, 10)
        # dec_out = EPR: # (bs*#massinf, pred_len, hidden_size=10)
        return dec_out, None if self.tm_seq2seq else dec_h, severity_rmseloss
    
class DecoderAttention(nn.Module):
    def __init__(self, config):
        super(DecoderAttention, self).__init__()
        
        self.hidden_size = config.seq2seq_encoder_lstm_cell
        self.directions = 2
        self.linear = nn.Linear(self.hidden_size*self.directions,
                                self.hidden_size*self.directions, bias=False)
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
        
        
    
class ViewCombiner(nn.Module):
    def __init__(self, config, dropout_p=0.1):
        super(ViewCombiner, self).__init__()
        
        self.region = config.r
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        self.pred_len = config.p
        self.hidden_size = 10
        self.device = config.device
        
        self.outer_product_combiner = config.outer_product_combiner
        self.dnn_combiner = config.dnn_combiner
        
        self.ab_ECR = config.ablation_ECR
        self.ab_GER = config.ablation_GER
        
        assert self.outer_product_combiner * self.dnn_combiner == 0
        
        if self.outer_product_combiner: # use outer product to combine
            if self.ab_ECR == False:
                self.ECR_linears = nn.ModuleList([nn.Linear(self.numOfIndustry*3, 
                                                            self.numOfIndustry) for _ in range(3)] +\
                                                 [nn.Linear(self.numOfIndustry*4, 
                                                            self.numOfIndustry)])
                self.ECR_layernorms = nn.ModuleList([nn.LayerNorm([self.numOfIndustry],  eps=1e-16) for _ in range(4)])
                self.MIR_layernorms = nn.LayerNorm([self.numOfMassInfection, 
                                                                   self.numOfIndustry,
                                                                   self.pred_len, 1], eps=1e-16)
    
                self.register_parameter('weight_sum',
                                        nn.Parameter(torch.ones(3)/3)) # 1/self.n_feats_buz_struct

            self.softmax = nn.Softmax(dim=-1)
            self.ECR_layernorm = nn.LayerNorm([self.numOfMassInfection, self.numOfIndustry])
            self.EPR_linear = nn.Linear(in_features=self.hidden_size, out_features= 1)
            
            self.EPR_GER_layernorm = nn.LayerNorm([self.numOfMassInfection, 
                                                     self.pred_len], eps=1e-16)
            
    def forward(self, ECR, GER, EPR, week_emb=None, buz_bias=None):
        """
        @ECR: ECR[i] (#region,#massinf,#ind) x 3
        @GER: (bs, #massinf)
        @EPR: (bs * #massinf, #pred, hidden=10)
        @week_emb: (bs, #ind, #pred)
        @buz_bias: (bs, #ind)
        
        option: 1.outer product or 2.lstm
        
        return MIR: (bs, #massinf, #ind, #pred)
        """
        bs = GER.size(0)
        num_repeat = bs//self.region # 4
            
        # 1.outer product
        if self.outer_product_combiner:
            
            if self.ab_ECR == False: 
                BS, CS, OS = ECR
                ECR_interactions = [BS*self.weight_sum[0]+CS*self.weight_sum[1]+OS*self.weight_sum[2]]
                # (#region,#massinf,#ind)
                
            EPR = self.EPR_linear(EPR.view(-1, self.hidden_size))
            EPR = EPR.view(-1, self.pred_len).view(bs, self.numOfMassInfection, self.pred_len) 
            #EPR: (bs, #massinf, #pred)

            # GER x EPR
            EPR_GER = EPR * GER.unsqueeze(-1) # (bs, #massinf, #pred) 
            
            
            # ECR x GER x EPR
            if self.ab_ECR: 
                MIR = self.EPR_GER_layernorm(EPR_GER)

            elif self.ab_GER: 
                MIR = [EPR.unsqueeze(2)*intrctn.repeat(num_repeat,1,1).unsqueeze(-1)
                       for intrctn in ECR_interactions]
                MIR = torch.stack(MIR, dim=-1)
                MIR = self.MIR_layernorms(MIR)
                
            else:
                MIR = [EPR_GER.unsqueeze(2)*intrctn.repeat(num_repeat,1,1).unsqueeze(-1)
                       for intrctn in ECR_interactions]
                MIR = torch.stack(MIR, dim=-1)
                MIR = self.MIR_layernorms(MIR)
                # ECR_EPR_GER: (bs, #massinf, 1, #pred) * (bs, #massinf, #ind, 1) = (bs, #massinf, #ind, #pred)
            
        return MIR
        

class MacroscopicAggregator(nn.Module):
    
    CONSTANT = 1 # for zero active mass infections
    
    def __init__(self, config):
        super(MacroscopicAggregator, self).__init__()
        
        self.region = config.r
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        
        self.pred_len = config.p
        
        self.EPR_hidden_size = 10
        EPR_linear_features = [self.EPR_hidden_size, (self.EPR_hidden_size)//2,1]
        self.EPR_linear = nn.ModuleList([nn.Sequential(nn.Linear(in_features=EPR_linear_features[0],
                                                                 out_features=EPR_linear_features[1]),
                                                       nn.ReLU()),  
                                         nn.Linear(in_features=EPR_linear_features[1],
                                                   out_features=EPR_linear_features[2])])
        self.ab_ECR = config.ablation_ECR
        self.ab_GER = config.ablation_GER
        self.ab_MAR = config.ablation_MAR
        ECR_dim = 0 if self.ab_ECR else 3
        self.MIR_dim = 1 
        GER_dim = 0 if self.ab_GER else 1
        
        self.input_size = self.MIR_dim
        Q_K_features = [self.input_size,self.input_size*2,self.input_size]
        self.fusion_linears_Q = nn.ModuleList([nn.Sequential(
                                        nn.Linear(in_features=Q_K_features[i], out_features=Q_K_features[i+1]),
                                        nn.ReLU()) for i in range(2)])
        
        self.fusion_linears_K = nn.ModuleList([nn.Sequential(
                                        nn.Linear(in_features=Q_K_features[i], out_features=Q_K_features[i+1]),
                                        nn.ReLU()) for i in range(2)])
        
        self.layernorm_Q_K = nn.LayerNorm(normalized_shape=[self.numOfMassInfection, self.numOfIndustry,
                                                            self.pred_len, self.input_size])
        
        self.attn_softmax = nn.Softmax(dim=-1)
        self.feat_linear =  nn.Linear(in_features=self.input_size, out_features=1)
        self.MAR_layernorm = nn.LayerNorm(normalized_shape=[self.numOfIndustry, self.pred_len])
        
        if self.ab_ECR:
            self.numOfIndustry = 1
            self.numOfIndustry_ECR = config.n
        self.register_parameter('MAR_metaweights',
                                nn.Parameter(torch.randn(self.region*self.numOfIndustry, 
                                                         self.pred_len*self.input_size))) 
        self.register_parameter('MAR_metabias',
                                nn.Parameter(torch.zeros(self.region*self.numOfIndustry,))) 
        nn.init.xavier_uniform_(self.MAR_metaweights)
        nn.init.constant_(self.MAR_metabias, 0.0)
        
        self.MIR_residual_linear = nn.Linear(in_features=7, out_features=1)
        self.MIR_attn_residual_linear = nn.Linear(in_features=7, out_features=1)
        
    def forward(self, MIR, ECR, GER, EPR, covid_mask):
        """
        @MIR: (bs, #massinf, #ind, #pred, 7)
            if ab_ECR: (bs, #massinf, #pred)
        @ECR: ECR[i] (#region,#massinf,#ind) x 3
        @GER: (bs, #massinf)
        @EPR: (bs* #massinf, #pred, hidden)
        @covid_mask: (bs, #massinf)
        
        return MAR: (bs, #ind, #pred)
        
        1. compute MIR_residual
        2. compute ATTN
        3. compute MAR = layernorm(MIR*ATTN + MIR_residual)
        """
        # 1. compute MIR_residual
        bs_active_mass_inf = covid_mask.sum(1) # (bs)
        bs_active_mass_inf[bs_active_mass_inf == 0] = self.CONSTANT # to avoid ZerodivisionError
        MIR_residual_weights = covid_mask / bs_active_mass_inf.unsqueeze(-1) # (bs, #massinf)
        
        if self.ab_ECR:
            MIR_residual = MIR * MIR_residual_weights.unsqueeze(-1) # (bs, #massinf, #pred)
            MIR_residual = MIR_residual.sum(1) # (bs, #pred)
            MIR_residual = MIR_residual.unsqueeze(1).repeat(1,self.numOfIndustry,1) # (bs, ind, #pred)
            
        else:
            MIR_residual = MIR * MIR_residual_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            MIR_residual = MIR_residual.sum(1) # (bs, #ind, #pred) or (bs, #ind, #pred, 7)
            MIR_residual = MIR_residual.mean(-1) # (bs, #ind, #pred)
            
        if self.ab_MAR: 
            return self.MAR_layernorm(MIR_residual)
        
        # 2. compute ATTN
        bs = bs_active_mass_inf.size(0)
        num_repeat = bs//self.region
        ################
        for layer in self.fusion_linears_Q:
            MIR = layer(MIR)    
            
        MIR = MIR.view(bs, self.numOfMassInfection, self.numOfIndustry, self.pred_len, self.input_size)
        MIR = MIR.transpose(1,2).contiguous().view((bs, self.numOfIndustry, 
                                                        self.numOfMassInfection, 
                                                        self.pred_len*self.input_size))
        MIR = MIR.view(bs* self.numOfIndustry, self.numOfMassInfection, -1)
        # MIR_Q: (bs* #ind, #mass, pred*input_sz)
        
        metaweights = self.MAR_metaweights.unsqueeze(0).expand(num_repeat,-1,-1).contiguous().view(bs*self.numOfIndustry, -1)
        metabias = self.MAR_metabias.unsqueeze(0).expand(num_repeat,-1).contiguous().view(bs*self.numOfIndustry)
        metaweights = metaweights.unsqueeze(1) # metaweights: (bs* #ind, 1, pred*input_sz)
        metabias = metabias.unsqueeze(-1) # metabias: (bs* #ind, 1)
        
        attn_weights = (MIR * metaweights).sum(-1) + metabias # attn_weights: (bs* #ind, #mass)
        covid_mask = covid_mask.unsqueeze(1).repeat(1, self.numOfIndustry, 1).contiguous()
        covid_mask = covid_mask.view(-1, self.numOfMassInfection) # (bs * #ind, #massinf)
        
        # no active massinf
        attn_for_No_activemassinf = attn_weights[covid_mask.sum(1) == 0,:] * 0  # (none, #mass)
        
        covid_mask_for_activemassinf = covid_mask[covid_mask.sum(1) != 0,:]
        attn_for_activemassinf = attn_weights[covid_mask.sum(1) != 0,:].masked_fill(covid_mask_for_activemassinf == 0,
                                                                            -float('inf'))
        attn_for_activemassinf = self.attn_softmax(attn_for_activemassinf) # (none, #mass)
        
        attn_weights[covid_mask.sum(1) == 0,:] = attn_for_No_activemassinf
        attn_weights[covid_mask.sum(1) != 0,:] = attn_for_activemassinf
        attn_weights = attn_weights.unsqueeze(-1)
        # attn_weights: (bs * #ind, #mass, 1)
        
        # 3. compute MAR = layernorm(MIR*ATTN + MIR_residual)

        MIR = MIR.view((bs* self.numOfIndustry, 
                            self.numOfMassInfection,
                            self.pred_len,
                            self.input_size)) # (bs*#ind, #mass, #pred, #feat)
        MIR = (MIR*attn_weights.unsqueeze(-1)).sum(1) # (bs*#ind, #pred, #feat)
        MIR = MIR.mean(-1) # (bs*#ind, #pred)
        MIR = MIR.view(bs, self.numOfIndustry, self.pred_len)
        if self.ab_ECR: 
            MIR = MIR.repeat(1,self.numOfIndustry_ECR,1) # (bs, ind, pred)
        MAR = self.MAR_layernorm(MIR + MIR_residual)
        
        
        return MAR, attn_weights # (bs, #ind, #pred), (bs * #ind, #mass, 1)
        

class WeekEmb(nn.Module):
    def __init__(self, config):
        super(WeekEmb, self).__init__()
        
        self.region = config.r
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        self.pred_len = config.p
        self.embedding_dim = config.e
        self.ablation_ECR = config.ablation_ECR
        
        self.weekemb_linear = nn.Sequential(nn.Linear(self.embedding_dim,self.embedding_dim) 
                                            if self.ablation_ECR else 
                                            nn.Linear(self.embedding_dim*2,self.embedding_dim),
                                            nn.ReLU(),
                                            nn.Linear(self.embedding_dim, 1),)
        self.weekemb_layernorm = nn.LayerNorm(normalized_shape=[self.pred_len, self.numOfIndustry],)
        
    def forward(self, weekdays_dec, district_buz_emb):
        """
        @weekdays_dec: (bs * #massinf, 1+pred_len, emb_dim = 6)
        @district_buz_emb: eco_view_enc.district_buz_emb.weight: (regions * ind, emb_dim = 20)
        
        return week_emb: (bs, #ind, #pred)
        """
        bs = weekdays_dec.size(0) // self.numOfMassInfection
        num_repeats = bs // self.region
        week_emb_dim = weekdays_dec.size(-1)
        district_buz_emb_dim = district_buz_emb.size(-1)
        
        weekdays_dec = weekdays_dec[:, 1:, :].view(bs, self.numOfMassInfection, 
                                                   self.pred_len, -1)[:,0] #(bs, pred_len, ind, emb_dim)
        weekdays_dec = weekdays_dec.unsqueeze(2).repeat(1,1,
                                                        self.numOfIndustry, 1) 
        # weekdays_dec: (bs, pred_len, ind, emb_dim=6)
        district_buz_emb = district_buz_emb.view(self.region, self.numOfIndustry, -1)
        district_buz_emb = district_buz_emb.unsqueeze(1).repeat(1, self.pred_len, 
                                                                1, 1)
        district_buz_emb = district_buz_emb.repeat(num_repeats, 1, 1, 1)
        # district_buz_emb: (bs, pred_len, ind, emb_dim=20)

        if self.ablation_ECR:
            week_emb = weekdays_dec.view(-1, week_emb_dim) #(bs* pred_len* ind, emb_dim)
            week_emb = self.weekemb_linear(week_emb).view(bs, self.pred_len, self.numOfIndustry)

        else:
            week_emb = torch.cat([weekdays_dec,district_buz_emb], dim=-1).view(-1, 
                                                                               week_emb_dim+district_buz_emb_dim)
            week_emb = self.weekemb_linear(week_emb).view(bs, self.pred_len, self.numOfIndustry)
            week_emb = self.weekemb_layernorm(week_emb) # week_emb: (bs, pred_len, ind)
        week_emb = week_emb.transpose(1,2).contiguous()
        
        return week_emb
    
    
    
    
    
    
    
    
class COVIDEENet(nn.Module):
    def __init__(self, config):
        super(COVIDEENet, self).__init__()
        
        self.config = config
        
        self.region = config.r
        self.targetPeriod = config.p
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.c
        self.num_heads = config.h
        self.embedding_dim = config.e
        self.activation = config.activation
        self.covid_start = config.covid_start
        # encoders ###########################################
        self.economic_view_subenc = EconomicViewSubEncoder(config)
        self.geographic_view_subenc = GeographicViewSubEncoder(config)
        self.epidemic_view_subenc = EpidemicViewSubEncoder(config)
        self.view_combiner = ViewCombiner(config)
        self.macroscopic_agg = MacroscopicAggregator(config)
        
        self.weekemb = WeekEmb(config)
        self.register_parameter('buz_bias', nn.Parameter(torch.randn(self.region, self.numOfIndustry)))
        self.last_linear = nn.Linear(in_features = self.targetPeriod,
                                     out_features = self.targetPeriod)
        ######################################################
        
        # ablation option ####################################
        self.ab_ECR = config.ablation_ECR
        self.ab_GER = config.ablation_GER
        self.ab_MAR = config.ablation_MAR
        ######################################################
        
        self.loss = torch.nn.MSELoss(reduction="none")
        
    def compute_y_hat(self, MAR, week_emb, buz_bias):
        """
        @MAR: (bs, ind, pred)
        @week_emb: (bs, ind, pred)
        @buz_bias: (ind, 1) or (reg,ind)
        
        return y_hat: (bs, ind, pred)
        """
        bs = MAR.size(0)
        rep = bs//self.region
        
        ## reg_ind_bias
        buz_bias = buz_bias.repeat(rep, 1).unsqueeze(-1) # (bs, ind, 1)
        if self.ab_ECR:
            buz_bias = buz_bias.mean(1).unsqueeze(-1)
            
        y_hat = MAR + week_emb + buz_bias

        return y_hat
    
    def forward(self, x, verbose=False, inspect=False):
        if inspect: 
            modeling_output = {}
            
        x, y, loss_mask, metadata = x
        
        business_structure_target, \
        business_structure_infected, \
        customer_structure_target, \
        customer_structure_infected, \
        index_target_idx, \
        index_infected_idx, \
        physical_distance, \
        contextual_distance, \
        covid_outbreak_business, \
        epidemiological_severity, \
        covid_elapsed_day, \
        weekdays, \
        mask, \
        covidMask = x
            
        # to gpu/cpu
        business_structure_target = business_structure_target.to(self.config.device)
        business_structure_infected = business_structure_infected.to(self.config.device)
        customer_structure_target = customer_structure_target.to(self.config.device)
        customer_structure_infected = customer_structure_infected.to(self.config.device)
        index_target_idx = index_target_idx.to(self.config.device)
        index_infected_idx = index_infected_idx.to(self.config.device)
        physical_distance = physical_distance.to(self.config.device)
        contextual_distance = contextual_distance.to(self.config.device)
        covid_outbreak_business = covid_outbreak_business.to(self.config.device)
        epidemiological_severity = epidemiological_severity.to(self.config.device)
        covid_elapsed_day = covid_elapsed_day.to(self.config.device)
        weekdays = weekdays.to(self.config.device)
        mask = mask.to(self.config.device) # (bs, #massinf, dates, 1): # days passed
        covidMask = covidMask.to(self.config.device, self.config.dtype) # (bs, #massinf, 1) : active massinfs
        # lossmaks :# (bs, #ind): (buz x region) pairs over thrh-sparsity
        bs = covidMask.size(0)
        
        y = y.to(self.config.device)
        
        # economic_view_encode ######################### : ECR
        if self.config.ablation_ECR:
            if self.config.dnn_combiner:
                ECR = (torch.zeros(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device),
                       torch.zeros(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device),
                       torch.zeros(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device))
            if self.config.outer_product_combiner:
                ECR = (torch.ones(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device),
                       torch.ones(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device),
                       torch.ones(self.region,self.numOfMassInfection,self.numOfIndustry).to(self.config.device))
        else:
            ECR = self.economic_view_subenc(business_structure_target, \
                                            business_structure_infected, \
                                            customer_structure_target, \
                                            customer_structure_infected, \
                                            index_target_idx, \
                                            index_infected_idx, \
                                            covid_outbreak_business, ) # ECR[i] (#region,#massinf,#ind) x 3
        
        # geographic_view_encode #########################: GER
        if self.config.ablation_GER:
            if self.config.dnn_combiner:
                GER = torch.zeros(bs, self.numOfMassInfection).to(self.config.device)
            if self.config.outer_product_combiner:
                GER = torch.ones(bs, self.numOfMassInfection).to(self.config.device)
        else:
            GER = self.geographic_view_subenc(physical_distance,
                                              contextual_distance) # (bs, c)
        
        # epidemic_view_encode ######################### : EPR
        EPR, _, severity_rmseloss, weekdays_dec = self.epidemic_view_subenc(epidemiological_severity, 
                                                                           covid_elapsed_day,
                                                                           weekdays,
                                                                           mask,
                                                                           covidMask,
                                                                           self.covid_start)  # (bs, c, #pred)
        
        # external_feat #########################
        week_emb = self.weekemb(weekdays_dec,
                                torch.zeros_like(self.economic_view_subenc.district_buz_emb.weight.clone())
                                if self.config.ablation_ECR else self.economic_view_subenc.district_buz_emb.weight.clone()) # (bs, #ind, #pred)
        buz_bias = self.buz_bias # (bs, #ind)
        
        # view_combiner #########################: MIR
        MIR = self.view_combiner(ECR, GER, EPR, week_emb, buz_bias) # (bs, c, #ind, #pred)
        # macroscopic_aggregate ######################### : MAR
        if self.config.ablation_MAR:
            MAR = self.macroscopic_agg(MIR, ECR, GER, EPR, covidMask)
        else :
            MAR, attn_weights = self.macroscopic_agg(MIR, ECR, GER, EPR, covidMask) # (bs, #ind, #pred)
            
        # compute y_hat #########################
        if inspect:
            
            y_hat = self.compute_y_hat(MAR, week_emb, buz_bias) 
            # y_hat = (bs, ind, pred)
            y_hat = self.last_linear(y_hat.view(-1, self.targetPeriod)).view(-1, self.numOfIndustry, self.targetPeriod)
            
            modeling_output["ECR"] = ([ECR[i].cpu() for i in range(len(ECR))])
            modeling_output["GER"] = GER.cpu()
            modeling_output["EPR"] = EPR.cpu()
            modeling_output["MIR"] = MIR.cpu()
            modeling_output["MAR"] = MAR.cpu()
            modeling_output["attn_weights"] = attn_weights.cpu() # (bs * #ind, #mass, 1)
            modeling_output["week_emb"] = week_emb.cpu()
            modeling_output["buz_bias"] = buz_bias.cpu()
            modeling_output["y_hat"] = y_hat.cpu()
            
        else :
            y_hat = self.compute_y_hat(MAR, week_emb, buz_bias) 
            y_hat = self.last_linear(y_hat.view(-1, self.targetPeriod)).view(-1, self.numOfIndustry, self.targetPeriod)
           
        # compute MSELoss ######################### 
        return (self.loss(y_hat, y.to(torch.float32)), y_hat, modeling_output)\
                if inspect else (self.loss(y_hat, y.to(torch.float32)), y_hat, None)
    
    
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None :
            nn.init.constant_(m.bias.data, 0.0)
            