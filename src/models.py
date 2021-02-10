#%%
#####################################################################
# Title  : Main Model Components
# Date   : 2020/11/19
# Update : 2021/02/08
# Author : Hyangsuk
#####################################################################
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import pickle, os, datetime

from modeling_outputs import COVIDNetOutputs, MainOutputs
###################################################################################################
# Model Components
###################################################################################################
# Economy-view Sub-encoder ########################################################################
class Attn(tf.keras.layers.Layer):
    def __init__(self, names = "Attn"):
        super(Attn, self).__init__()
        self.names = names
    def build(self, batch_input_shape):
        self.W1 = self.add_weight(
            name = self.names, shape = [batch_input_shape[-1], batch_input_shape[-1]],
            initializer = 'glorot_normal', dtype = tf.dtypes.float32, trainable = True)
        self.W2 = self.add_weight(
            name = self.names, shape = [batch_input_shape[-1], batch_input_shape[-1]],
            initializer = 'glorot_normal', dtype = tf.dtypes.float32, trainable = True)
        super().build(batch_input_shape)
    def call(self, E):
        # E [None, vocab, hidden], W = [hidden, hidden]
        # E@W = [None, vocab, hidden], (E@W).T = [None, hidden, vocab]
        # results = [None, vocab, vocab]
        return tf.nn.softmax(
                        tf.raw_ops.BatchMatMulV2(
                            x = E @ self.W1,
                            y = tf.transpose(E @ self.W2, perm=(0,2,1))) / tf.sqrt(tf.constant(E.shape[-1], dtype = tf.dtypes.float32)),
                         axis = 1)
# Economy-view Sub-encoder ########################################################################

# Utils ###########################################################################################
class Weighted_sum(tf.keras.layers.Layer):
    def __init__(self, names = "weights"):
        super(Weighted_sum, self).__init__()
        self.names = names
    def build(self, batch_input_shape):
        self.W = self.add_weight(
            name = self.names, shape = [batch_input_shape[-1], 1],
            initializer = 'glorot_normal', dtype = tf.dtypes.float32, 
            constraint =  tf.keras.constraints.non_neg(),
            # constraint = lambda x: tf.clip_by_value(x, 0, 1), 
            trainable = True )
        super().build(batch_input_shape)
    def call(self, inputs): # inputs : M_target
        return inputs @ tf.nn.softmax(self.W)
# Utils ###########################################################################################
    
class BuzCusStructureSim(tf.keras.Model):
    def __init__(self, num_heads):
        super(BuzCusStructureSim, self).__init__()
        self.num_heads = num_heads

        self.weighted_sum_BusinessStructure = Weighted_sum(names = "W_BusinessStructure_weight")
        self.weighted_sum_CustomerStructure = Weighted_sum(names = "W_CustomerStructure_weight")
                                            
        self.BS = [Attn(names = "Attn{}".format(i)) for i in range(self.num_heads)]
        self.BSDot = tf.keras.layers.Dot(axes=(1,1), name = "BSCS")

        self.cosine = tf.keras.losses.CosineSimilarity(axis = -1, reduction=tf.keras.losses.Reduction.NONE )
        self.kld = tf.keras.losses.KLDivergence(reduction = tf.keras.losses.Reduction.NONE )
        
        #############################################################################################
        self.BSLayerNorm = tf.keras.layers.LayerNormalization(epsilon = 1e-16, name = "BSLayerNorm")
        self.CSLayerNorm = tf.keras.layers.LayerNormalization(epsilon = 1e-16, name = "CSLayerNorm")
        #############################################################################################

        self.elmwise = tf.keras.layers.Multiply(name = "elmwise_BuzCusStructureSim")
        self.W = Weighted_sum(names = "W_BuzCusStructureSim_weight")

    def call(self, B_target, B_infected, E_target, E_infected, C_target, C_infected):
        # Initialize embedding vector first

        # B_target, B_infected, C_target, C_infected, = inputs
        B_target = self.weighted_sum_BusinessStructure(B_target)
        B_target = tf.squeeze(B_target, axis = -1)

        B_infected = self.weighted_sum_BusinessStructure(B_infected)
        B_infected = tf.squeeze(B_infected, axis = -1)

        # Similarity of Major Industry
        # B_target : [None, config.n = vocab], ss(E_target) : [None, vocab, vocab] => [None, vocab]
        BR_target = tf.transpose(tf.stack([ self.BSDot([B_target, ss(E_target)]) for ss in self.BS], axis=1), perm=((0,2,1))) # [None, vocab, num_heads]
        BR_infected = tf.transpose(tf.stack([ self.BSDot([B_infected, ss(E_infected)]) for ss in self.BS], axis=1), perm=((0,2,1))) # [None, vocab, num_heads]
        BS = self.BSLayerNorm(self.cosine(BR_infected, BR_target)) # [None, config.n]
        
        # Similarity of Customer Distribution
        ## JSD
        C_target = tf.squeeze(self.weighted_sum_CustomerStructure(C_target), axis = -1)
        C_infected = tf.squeeze(self.weighted_sum_CustomerStructure(C_infected), axis = -1)

        C = 0.5 * (C_target + C_infected)
        C = 0.5 * self.kld(C_infected, C) + 0.5 * self.kld(C_target, C)
        CS = self.CSLayerNorm(C) # [None, config.n]
        
        composite = tf.stack([BS,CS], axis=-1) 
        outputs = self.W(composite) 
        outputs = tf.squeeze(outputs, axis = -1) 
        
        return outputs, BS, CS, 

class OSLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(OSLayer, self).__init__()
    def build(self, batch_input_shape):
        self.W = self.add_weight(
            name = "W_OS", shape = [batch_input_shape[-1], batch_input_shape[-1]],
            initializer = 'glorot_normal', dtype=tf.dtypes.float32, trainable = True)
        super().build(batch_input_shape)
    def call(self, E_target):
        # E_target : [None, vocab, hidden]
        return tf.keras.activations.tanh(E_target @ self.W)

class OutbreakBusinessSim(tf.keras.Model):
    def __init__(self):
        super(OutbreakBusinessSim,self).__init__()
        self.OS = OSLayer()
        self.OSDot = tf.keras.layers.Dot(axes=(-1,-1), name = "OSDot")

    def call(self, E_target, E_infected, outbreak_business):
        # E_target & E_infected : [None, config.n, hidden], 
        # outbreak_business : [None, 1, 1]
        # self.S(E) : [None, 35, 100]
        outbreak_business = tf.concat([tf.reshape(tf.range(start = 0, limit = outbreak_business.shape[0], dtype = tf.dtypes.int64), shape=(-1,1)), 
                                        tf.cast(outbreak_business, dtype=tf.dtypes.int64)], axis = -1)
        outbreak_business_emb = tf.gather_nd(E_infected, tf.cast(outbreak_business, dtype = tf.dtypes.int32)) # [None, hidden]
        OS = self.OSDot([outbreak_business_emb, self.OS(E_target)]) # [None, hidden] . [None, config.n, hidden] = [None, config.n]
        return OS, outbreak_business_emb
# Economy-view Sub-encoder ##########################################################################

# Geography-view Sub-encoder ########################################################################
class GeographyViewSubEncoder(tf.keras.Model):
    def __init__(self, units, activation = 'sigmoid'):
        super(GeographyViewSubEncoder, self).__init__()
        self.GeographyViewSubEncoderFCN = tf.keras.layers.Dense(units, name = "GeographyViewSubEncoderFCN")
        self.GeographyViewSubEncoderBatchNorm = tf.keras.layers.BatchNormalization(epsilon=1e-16, name = "GeographyViewSubEncoderBatchNorm")
        self.ac = tf.keras.activations.get(activation)

    def call(self, inputs, training):
        physical_distance, contextual_distance = inputs
        pc_distance = tf.concat([physical_distance, contextual_distance], axis = -1)
        outputs = self.GeographyViewSubEncoderBatchNorm(self.GeographyViewSubEncoderFCN(pc_distance), training = training)
        return self.ac(outputs), physical_distance, contextual_distance
# Geography-view Sub-encoder ##########################################################################

# Epidemic-view Sub-encoder ###########################################################################
class Encoder(tf.keras.Model):
    def __init__(self, lstm_cell):
        super(Encoder, self).__init__()
        self.lstm_cell = lstm_cell
        self.lstm = tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(self.lstm_cell), return_sequences = True, return_state = True)

    def call(self, inputs, mask, training = True):
        encoder_outputs = self.lstm(inputs, mask = mask, training = training)
        return encoder_outputs

class Decoder(tf.keras.Model):
    def __init__(self, lstm_cell, targetPeriod, activation):
        super(Decoder, self).__init__()
        self.lstm_cell = lstm_cell
        self.targetPeriod = targetPeriod
        self.lstm = tf.keras.layers.RNN(tfa.rnn.LayerNormLSTMCell(self.lstm_cell), return_sequences = True, return_state = True)
        self.decoderFCN = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(4, activation = activation, name = "decoderFCN"), name = "decoderFCN_TimeDistributed")

    def call(self, inputs, encoder_outputs, mask, training = True):
        if training:
            decoder_inputs = inputs
            decoder_outputs = self.lstm(decoder_inputs, initial_state = encoder_outputs[1:], training = training)
            decoder_outputs = self.decoderFCN(decoder_outputs[0])
            return decoder_outputs
        else :
            outputs = []
            decoder_inputs = tf.expand_dims(inputs[:,0,:], axis = 1)
            decoder_hiddens = encoder_outputs[1:] 
            for t in range(self.targetPeriod):
                decoder_outputs = self.lstm(decoder_inputs, initial_state = decoder_hiddens, training = training)
                decoder_inputs = self.decoderFCN(decoder_outputs[0])   
                decoder_hidden = decoder_outputs[1:]
                outputs.append(decoder_inputs)
            outputs = tf.concat(outputs, axis = 1)
            return outputs

class Seq2Seq(tf.keras.Model):
    def __init__(self, encoder_lstm_cell, decoder_lstm_cell, targetPeriod, activation):
        super(Seq2Seq, self).__init__()
        self.encoder_lstm_cell = encoder_lstm_cell
        self.decoder_lstm_cell = decoder_lstm_cell
        self.targetPeriod = targetPeriod
        self.encoder = Encoder(lstm_cell = self.encoder_lstm_cell)
        self.decoder = Decoder(lstm_cell = self.decoder_lstm_cell, targetPeriod = self.targetPeriod, activation = activation)
        
    def call(self, inputs, mask, training):
        encoder_inputs, decoder_inputs = inputs[:,:-self.targetPeriod, :], inputs[:, -self.targetPeriod:, :]
        mask = mask[:,:-1]
        encoder_outputs = self.encoder(encoder_inputs, mask, training = training)
        decoder_outputs = self.decoder(decoder_inputs, encoder_outputs, mask, training = training)
        return decoder_outputs

class EpidemicViewSubEncoder(tf.keras.Model):
    def __init__(self, encoder_lstm_cell, decoder_lstm_cell, targetPeriod, activation):
        super(EpidemicViewSubEncoder, self).__init__()
        self.targetPeriod = targetPeriod
        self.seq2seq = Seq2Seq(encoder_lstm_cell = encoder_lstm_cell, decoder_lstm_cell = decoder_lstm_cell, targetPeriod = self.targetPeriod, activation = activation)

    def call(self, inputs, mask, training):
        outputs = self.seq2seq(inputs = inputs, mask = mask, training = training)
        return outputs # [None, targetPeriod, units]
# Epidemic-view Sub-encoder ############################################################################

# External Features ####################################################################################
class WeekEmbedding(tf.keras.Model):
    def __init__(self, units, embedding_dim, numOfIndustry, activation):
        super(WeekEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.numOfIndustry = numOfIndustry

        if isinstance(units, list):
            self.units = units
        else:
            self.units = [units]

        self.WeekdayEmb = tf.keras.layers.Embedding(7, self.embedding_dim, dtype = tf.dtypes.float32, name = "WeekdayEmb")
        self.WeekdayEmbLayerNorm = tf.keras.layers.LayerNormalization(epsilon = 1e-16, name = "WeekdayEmbLayerNorm")

        self.WeekdayFCN = [tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(unit, activation = self.activation, name = "WeekdayFCN{}".format(unit)), name = "TimeDistributed_WeekdayFCN{}".format(unit)) for unit in self.units]
        self.WeekdayFCNLayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "WeekdayFCNLayerNorm")
    def call(self, inputs, EmbeddingTarget):
        ##################################################################################################
        # Create Week Embedding
        ##################################################################################################
        WeekdayEmb = self.WeekdayEmbLayerNorm(self.WeekdayEmb(tf.reshape(tf.tile(tf.reshape(tf.range(0,7),(7,1)),(inputs.shape[0],1)), (inputs.shape[0],7)))) # [None, 7, 20]
        WeekdayRepeated = tf.tile(WeekdayEmb,(1,self.numOfIndustry, 1)) # [None, 7 * config.n, 20]

        EmbeddingTargetRepeated = tf.repeat(EmbeddingTarget, repeats = [7]*self.numOfIndustry, axis = 1) # [None, 7 * config.n, 20]
        WeekIndustryEmbedding = tf.concat([WeekdayRepeated, EmbeddingTargetRepeated], axis = -1)

        for layer in self.WeekdayFCN:
            WeekIndustryEmbedding = layer(WeekIndustryEmbedding)
        WeekIndustryEmbedding2 = tf.squeeze(WeekIndustryEmbedding, axis = -1)
        WeekIndustryEmbedding3 = self.WeekdayFCNLayerNorm(tf.reshape(WeekIndustryEmbedding2, (-1, self.numOfIndustry, 7))) # (None, config.n, 7)
        WeekIndustryEmbedding4 = tf.transpose(WeekIndustryEmbedding3, perm = (0,2,1)) # (None, 7, config.n)

        ###################################################################################################
        # Rearrange the orders
        ###################################################################################################
        batchNum = tf.reshape(tf.repeat(tf.range(inputs.shape[0], dtype=tf.dtypes.int64), 
                            repeats = [inputs.shape[1]] * inputs.shape[0]), (-1, inputs.shape[1], 1))
        weekday = tf.concat([batchNum, tf.cast(inputs, dtype=tf.dtypes.int64)], axis = -1) # [None,7,2]

        return tf.gather_nd(WeekIndustryEmbedding4, weekday) # [None, config.w, config.n]

class Business_bias(tf.keras.layers.Layer):
    def __init__(self, names = "BusinessBias"):
        super(Business_bias, self).__init__()
        self.names = names
    def build(self, batch_input_shape):
        b_init = tf.random_normal_initializer()
        self.B = tf.Variable(
            initial_value = b_init(shape = (batch_input_shape[1], batch_input_shape[2]), dtype = tf.dtypes.float32), 
            trainable = True )
        super().build(batch_input_shape)
    def call(self, inputs): # inputs : M_target
        return tf.add(inputs, self.B)
# External Features ########################################################################################

############################################################################################################
# Main Model
############################################################################################################
class COVIDEENet(tf.keras.Model):
    def __init__(self, config):
        super(COVIDEENet, self).__init__()

        ####################################################################################################
        self.region = config.r
        self.targetPeriod = config.w
        self.numOfIndustry = config.n
        self.numOfMassInfection = config.m
        self.num_heads = config.h
        self.embedding_dim = config.e
        self.activation = config.activation
        ####################################################################################################

        # parameters #######################################################################################
        self.geographyViewFCN = config.geographyViewFCN

        self.seq2seq_encoder_lstm_cell = config.seq2seq_lstm_cell
        self.seq2seq_decoder_lstm_cell = config.seq2seq_lstm_cell
        self.epidemicViewFCN = config.epidemicViewFCN

        self.macroscopicAggFCN = config.w
        # parameters #######################################################################################

        # Ablation #########################################################################################
        self.ablation_economicView = config.ablation_economicView
        self.ablation_geographyView = config.ablation_geographyView
        self.ablation_macroscopicAgg = config.ablation_macroscopicAgg
        # Ablation #########################################################################################
 
        # Embedding ########################################################################################
        self.Emb = [tf.keras.layers.Embedding(self.numOfIndustry, self.embedding_dim, 
                                            dtype = tf.dtypes.float32, name = "regionEmb{}".format(i)
                                            ) for i in range(self.region)]
        self.regionEmbLayerNorm = tf.keras.layers.LayerNormalization(epsilon = 1e-16, axis=(1,2), name = "regionEmbLayerNorm")
        # Embedding #########################################################################################

        if self.ablation_economicView:
            # BuzCusStructureSimilarity #####################################################################
            self.BuzCusStructureSim = BuzCusStructureSim(num_heads = self.num_heads)
            #################################################################################################

            # OS ############################################################################################
            self.OutbreakBusinessSim = OutbreakBusinessSim()
            #################################################################################################

        if self.ablation_geographyView:
            # GER ###########################################################################################
            self.GeographyViewSubEncoder = GeographyViewSubEncoder(units = self.geographyViewFCN, activation = 'sigmoid')
            #################################################################################################

        # SEVERITY ##########################################################################################
        self.EpidemicViewSubEncoder = EpidemicViewSubEncoder(
                                                encoder_lstm_cell = self.seq2seq_encoder_lstm_cell,
                                                decoder_lstm_cell = self.seq2seq_decoder_lstm_cell,
                                                targetPeriod = self.targetPeriod, 
                                                activation = self.activation) # [None, targetPeriod, units]
        self.EpidemicViewSubEncoderFCN = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.epidemicViewFCN, name = "EpidemicViewSubEncoderFCN", use_bias = False), name = "TimeDistributed_EpidemicViewSubEncoderFCN")
        self.EpidemicViewSubEncoderLayerNorm = tf.keras.layers.LayerNormalization(axis = -2, epsilon = 1e-16, name = "EpidemicViewSubEncoderLayerNorm") # shorTermSeverityFCN [config.m, None, 7, 1] # OK
        self.EpidemicViewSubEncoderActivation = tf.keras.activations.get('sigmoid')
        #####################################################################################################
        
        # UTILS #############################################################################################
        self.elmwise = tf.keras.layers.Multiply(name = "elmwise_composite_lst")
        self.matmul = tf.keras.layers.Dot(axes=(2,1), name = "composite_lst_Dot")
        self.add = tf.keras.layers.Add(name = "composite_lst_add")
        #####################################################################################################

        # Composite #########################################################################################
        if self.ablation_economicView & self.ablation_geographyView :
            self.C1LayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "C1LayerNorm")
            self.C2LayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "C2LayerNorm")
            self.C3LayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "C3LayerNorm")
        self.alpha = tf.Variable(initial_value = 0.5, trainable = True, name = "W_SISO_weights", dtype = tf.dtypes.float32, constraint = tf.keras.constraints.non_neg())

        self.ViewCombinerLayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "ViewCombinerLayerNorm")

        if self.ablation_macroscopicAgg:
            # Event Attention ################################################################################
            self.macroscopicAggAttnQueryFCN = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(5, activation = self.activation, name = "macroscopicAggAttnQueryFCN"), name = "macroscopicAggAttnQueryFCN")
            self.macroscopicAggAttnDot = tf.keras.layers.Dot(axes=(-1,-1), name = "macroscopicAggAttnDot")
            self.macroscopicAggAttnSoftmax = tf.keras.layers.Softmax(axis = 1)
            ##################################################################################################
        self.MARSoftmax = tf.keras.layers.Softmax(axis = 1, name = "MARSoftmax")
        self.MARLayerNorm = tf.keras.layers.LayerNormalization(axis = (1,2), epsilon = 1e-16, name = "MARLayerNorm")

        # External Feature ###################################################################################
        self.WeekEmbedding = WeekEmbedding(units = [3,1], embedding_dim = self.embedding_dim, 
                                            activation = self.activation, numOfIndustry = self.numOfIndustry)
        # External Feature ###################################################################################

        ######################################################################################################
        self.MARWeekEmbFCN = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.targetPeriod, name = "MARWeekEmbFCN"), name = "TimeDistributed_MARWeekEmbFCN") # , activation = self.activation,
        ######################################################################################################
        
        # External Feature ###################################################################################
        self.Business_bias = Business_bias(names='BusinessBias')
        # External Feature ###################################################################################
#%%
    def call(self, inputs, training = True):
        business_structure_target, \
        business_structure_infected, \
        customer_structure_target, \
        customer_structure_infected, \
        index_target_idx, \
        index_infected_idx, \
        physical_distance, \
        contextual_distance, \
        covid_outbreak_business, \
        epidemiological_severity,\
        covid_elapsed_day, \
        weekdays, \
        mask, \
        covidMask = inputs

        weekdays_0 = weekdays[0]
        covidMask_0 = tf.transpose(covidMask, perm=(1,0,2)) # [config.m, batch, 1] -> [batch, config.m, 1]

        #######################################################################################################
        # Models for mass infections
        #######################################################################################################
        composite_outputs_allPred = []
        ComponentOutputs_allPred = []
        RegionEmbedding = self.regionEmbLayerNorm(tf.stack([E(tf.range(0,self.numOfIndustry)) for E in self.Emb], axis = 0))

        # COVIDEENet Sub Network ##############################################################################
        
        # Severity ############################################################################################
        sub_covid_inputs = tf.concat([epidemiological_severity, covid_elapsed_day[:,:,:-1,:]], axis = -1)
        e_batch = sub_covid_inputs.shape[1]
        e_length = sub_covid_inputs.shape[2]
        e_features = sub_covid_inputs.shape[3]
        sub_covid_inputs = tf.reshape(sub_covid_inputs, shape = (self.numOfMassInfection* e_batch, e_length, e_features)) # [config.m * batch, length, features]
        
        m_batch = mask.shape[1]
        m_length = mask.shape[2]
        sub_mask = tf.cast(tf.reshape(tf.squeeze(mask, axis = -1), (self.numOfMassInfection * m_batch, m_length)), dtype = tf.dtypes.bool)
        
        EPR__ = self.EpidemicViewSubEncoder(sub_covid_inputs, sub_mask, training = training) # (None, 7, 3)
        EPR_ = self.EpidemicViewSubEncoderFCN(EPR__)
        EPR_ = tf.reshape(EPR_, shape = (self.numOfMassInfection, e_batch, self.targetPeriod, 1)) # [config.m, None, config.w, 1]
        EPR_All = self.EpidemicViewSubEncoderActivation(self.EpidemicViewSubEncoderLayerNorm(EPR_))    # (config.m, None, config.w, 1)
        EPR_All_t = tf.transpose(EPR_All, perm = (1,0,2,3)) # [None, config.m, config.w, 1]
        # Severity ##############################################################################################

        for i in range(self.numOfMassInfection):
            # Inputs ############################################################################################
            sub_covid_elapsed_day = covid_elapsed_day[i]
            sub_business_structure_target = business_structure_target[i]
            sub_business_structure_infected = business_structure_infected[i]
            sub_customer_structure_target = customer_structure_target[i]
            sub_customer_structure_infected = customer_structure_infected[i]
            sub_index_target_idx = tf.cast(tf.squeeze(tf.squeeze(index_target_idx[i], axis = -1), axis = -1), dtype = tf.dtypes.int32)
            sub_index_infected_idx = tf.cast(tf.squeeze(tf.squeeze(index_infected_idx[i], axis = -1), axis = -1), dtype = tf.dtypes.int32)
            
            sub_E_target = tf.gather(RegionEmbedding, sub_index_target_idx)
            sub_E_infected = tf.gather(RegionEmbedding, sub_index_infected_idx)

            sub_physical_distance = tf.squeeze(physical_distance[i], axis = 1)
            sub_contextual_distance = tf.squeeze(contextual_distance[i], axis = 1)
            sub_covid_outbreak_business = tf.squeeze(covid_outbreak_business[i], axis = -1)
            sub_economic_severity = EPR_All[i] # (None, config.w, 1)
            # Inputs ##############################################################################################
                        
            if self.ablation_economicView:
                # BuzCusStructureSimilarity #######################################################################
                BuzCusStructureSimilarity, BS, CS = self.BuzCusStructureSim(
                    sub_business_structure_target,
                    sub_business_structure_infected,
                    sub_E_target,
                    sub_E_infected,
                    sub_customer_structure_target,
                    sub_customer_structure_infected,
                )
                # BuzCusStructureSimilarity #######################################################################

                # OS ##############################################################################################
                OS, OutbreakBusinessEmb = self.OutbreakBusinessSim(sub_E_target, sub_E_infected, sub_covid_outbreak_business)
                # OS ##############################################################################################
            else :
                BuzCusStructureSimilarity = tf.ones((sub_business_structure_target.shape[0], self.numOfIndustry), dtype = tf.dtypes.float32)
                BS = tf.constant([0], dtype = tf.dtypes.float32) # tf.ones((sub_business_structure_target.shape[0], self.numOfIndustry))
                CS = tf.constant([0], dtype = tf.dtypes.float32) #tf.ones((sub_business_structure_target.shape[0], self.numOfIndustry))

                OS = tf.ones((sub_business_structure_target.shape[0], self.numOfIndustry), dtype = tf.dtypes.float32)
                OutbreakBusinessEmb = tf.constant([0], dtype = tf.dtypes.float32)                
            
            if self.ablation_geographyView:
                # GeographyViewSubEncoder #########################################################################
                GER, P_Dist, C_Dist = self.GeographyViewSubEncoder((sub_physical_distance, sub_contextual_distance), training = training)
                # GeographyViewSubEncoder #########################################################################
            else:
                P_Dist = sub_physical_distance
                C_Dist = sub_contextual_distance
                GER = tf.ones_like(sub_physical_distance, dtype = tf.dtypes.float32)

            # Cross Features #######################################################################################
            if self.ablation_economicView & self.ablation_geographyView :
                composite_lst = tf.stack([ #
                                self.C1LayerNorm(self.matmul([sub_economic_severity, tf.expand_dims(BuzCusStructureSimilarity * GER, axis = 1)])),
                                self.C2LayerNorm(self.matmul([sub_economic_severity, tf.expand_dims(OS * GER, axis = 1)])),
                                self.C3LayerNorm(self.matmul([sub_economic_severity, tf.expand_dims(self.add([self.alpha * BuzCusStructureSimilarity, (1-self.alpha) * OS]) * GER, axis=1)]))
                                ], axis = -1) # [None, 7, config.n, 3]
            else: 
                composite_lst = tf.stack([ #
                                self.matmul([sub_economic_severity, tf.expand_dims(BuzCusStructureSimilarity * GER, axis = 1)]),
                                self.matmul([sub_economic_severity, tf.expand_dims(OS * GER, axis = 1)]),
                                self.matmul([sub_economic_severity, tf.expand_dims(self.add([self.alpha * BuzCusStructureSimilarity, (1-self.alpha) * OS]) * GER, axis=1)])
                                ], axis = -1) # [None, 7, config.n, 3]
            # Cross Features #########################################################################################

            composite_outputs = self.ViewCombinerLayerNorm(tf.math.reduce_mean(composite_lst, axis = -1)) # [None, config.w, config.n, 3] -> # [None, config.w, config.n]
            
            # Saving Outputs ##########################################################################################
            ComponentOutputs = COVIDNetOutputs(
                BS = BS,
                CS = CS,
                BuzCusStructureSimilarity = BuzCusStructureSimilarity,
                EmbeddingTarget = sub_E_target,
                EmbeddingInfected = sub_E_infected,
                OS = OS,
                OutbreakBusinessEmb = OutbreakBusinessEmb,
                P_Dist = P_Dist,
                C_Dist = C_Dist,
                GER = GER,
                EPR = sub_economic_severity, 
                composite_lst = composite_lst,
                composite_outputs = composite_outputs,
            )
            # Saving Outputs ###########################################################################################

            # ETC ######################################################################################################
            composite_outputs_allPred.append(composite_outputs) # [config.m, None, config.w, config.n]
            ComponentOutputs_allPred.append(ComponentOutputs)
            # ETC ######################################################################################################

        # COVIDEENet Sub Network #######################################################################################
        MIR = tf.stack(composite_outputs_allPred, axis = -1)        # (None, config.w, config.n, config.m)
        length = MIR.shape[0]
        
        # Masking, covidMask_0 : (None, config.m, 1)
        MIRMeanWeights1 = tf.where(tf.equal(covidMask_0, 0.0), 
                                    x = tf.constant([-float('inf')], dtype = tf.dtypes.float32),
                                    y = covidMask_0)
        MIRMeanWeights2 = self.MARSoftmax(MIRMeanWeights1)
        MIRMeanWeights3 = tf.where(tf.math.is_nan(MIRMeanWeights2), 
                                    x = tf.constant([0], dtype=tf.dtypes.float32), 
                                    y = MIRMeanWeights2)
        MIRResidual_ = tf.raw_ops.BatchMatMulV2(
                                    x = tf.reshape(MIR, (length, self.targetPeriod * self.numOfIndustry, self.numOfMassInfection)),
                                    y = MIRMeanWeights3)
        MIRResidual = tf.squeeze(tf.reshape(MIRResidual_, (length, self.targetPeriod, self.numOfIndustry, -1)), axis = -1)
        
        if self.ablation_macroscopicAgg:
            MAR = tf.concat([tf.transpose(MIR, perm = (0,3,1,2)), EPR_All_t], axis = -1) # (None, config.m, config.w, config.n + 1)
            macroscopicAggAttn__ = tf.reshape(MAR, (length, self.numOfMassInfection, -1)) # (None, config.m, config.w * (config.n + 1)) or (None, config.m, config.w * config.n))
            # Masking 
            macroscopicAggAttn_ = self.elmwise([macroscopicAggAttn__, covidMask_0]) # (None, config.m, config.w * (config.n + 1))
            macroscopicAggAttn_q = self.macroscopicAggAttnQueryFCN(macroscopicAggAttn_) # (None, config.m, 5)
            macroscopicAggAttn1 = self.macroscopicAggAttnDot([macroscopicAggAttn_q, macroscopicAggAttn_q])  / tf.sqrt(tf.constant(5, dtype = tf.dtypes.float32))# (None, config.m, config.m)
            macroscopicAggAttn2 = self.elmwise([macroscopicAggAttn1, covidMask_0])
            macroscopicAggAttn3 = tf.where(tf.equal(macroscopicAggAttn2, 0.0),
                                                    x = tf.constant([-float('inf')], dtype = tf.dtypes.float32),
                                                    y = macroscopicAggAttn2)
            macroscopicAggAttn4 = self.macroscopicAggAttnSoftmax(macroscopicAggAttn3)
            macroscopicAggAttn5 = tf.where(tf.math.is_nan(macroscopicAggAttn4), x = tf.constant([0], dtype = tf.dtypes.float32), y = macroscopicAggAttn4) # (None, config.m, config.m)
            MIRReshape = tf.reshape(MIR, ( -1, self.targetPeriod * self.numOfIndustry, self.numOfMassInfection )) # (None, config.w, config.n, config.m) -> (None, config.w * config.n, config.m)

            MAR__ = tf.reshape(tf.raw_ops.BatchMatMulV2( x = MIRReshape, 
                                                         y = macroscopicAggAttn5), 
                                                         shape = ( -1, self.targetPeriod, self.numOfIndustry, self.numOfMassInfection ))   # (None, config.w, config.n, config.m)
            MAR_ = tf.math.reduce_mean(MAR__, axis = -1) # (None, config.w, config.n)
            MAR = MAR_ + MIRResidual
        else :
            macroscopicAggAttn5 = tf.constant([0], dtype = tf.dtypes.float32)
            MAR = MIRResidual

        MAR = self.MARLayerNorm(MAR)

        WeekEmb = self.WeekEmbedding(weekdays_0,sub_E_target)                                # (None, config.w, config.n)

        if self.ablation_economicView:
            WeekEmb = WeekEmb
        else :
            WeekEmb = tf.expand_dims(tf.reduce_mean(WeekEmb, axis = -1), axis = -1)
        
        if self.ablation_geographyView:
            WeekEmb = WeekEmb
        else :
            WeekEmb = tf.expand_dims(tf.reduce_mean(WeekEmb, axis = 0), axis = 0)

        MARWeekEmb = MAR + WeekEmb    # (None, 7, config.n)
        
        MARWeekEmbFCN = self.MARWeekEmbFCN(tf.transpose(MARWeekEmb, perm = (0, 2, 1))) # (None, config.n , config.w)
        CovidImpact_ = self.Business_bias(MARWeekEmbFCN)
        
        # Calculate Covid Impact #########################################################################################
        CovidImpact = CovidImpact_  # [None, numOfIndustry, targetPeriod]
        # Calculate Covid Impact #########################################################################################
        
        MainOutputStorage = MainOutputs(
            ComponentOutputs = ComponentOutputs_allPred, 
            MARAttnWeight = macroscopicAggAttn5,
            MAR = MAR, 
            WeekEmb = WeekEmb,
            MARWeekEmb = MARWeekEmb, 
            MARWeekEmbFCN = MARWeekEmbFCN, # longTermInsensitivityWeekEmb,
            CovidImpact = CovidImpact,
            )
        
        return CovidImpact, MainOutputStorage  # [# of mass infection, batch, targetPeriod, 3] => [batch, # of mass infection, targetPeriod, 3]

# %%
