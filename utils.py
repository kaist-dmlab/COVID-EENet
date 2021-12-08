import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers
import pickle, glob, os
from dateutil.relativedelta import relativedelta
from Config import Config
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
plt.rcParams["font.family"] = "nanummyeongjo"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12.
plt.rcParams["xtick.labelsize"] = 12.
plt.rcParams["ytick.labelsize"] = 12.
plt.rcParams["axes.labelsize"] = 12.

import warnings
warnings.filterwarnings(action='ignore')

def rolling(data, w = 1):
    results = np.zeros_like(data)
    for i in range(data.shape[1]):
        if i < w:
            results[:,i] = np.concatenate([data[:, 0:i], data[:,i][..., np.newaxis], data[:, (i + 1):(i + 1 + w)]], axis = 1).sum(axis = 1) / (i + w + 1)

        elif i >= data.shape[1] - w:
            results[:,i] = np.concatenate([data[:, (i - w):i], data[:,i][..., np.newaxis], data[:, (i + 1):]], axis = 1).sum(axis = 1) / (data.shape[1] - i - 1 + w + 1)

        else :
            results[:,i] = np.concatenate([data[:, (i - w):i], data[:,i][..., np.newaxis], data[:, (i + 1):(i + 1 + w)]], axis = 1).sum(axis = 1) / (2 * w + 1)
    return results
    
def interpolation_na(row, duration, threshold):
    sparsity = row[:duration].isna().sum() / len(row[:duration])
    interpolation = sparsity < threshold
    if interpolation :
        # print("interpolation {}, sparsity {:3f}%".format(row.name, sparsity))
        isna = row.isna()
        for i in range(len(isna)):
            if isna[i]: # numpy.float32
                if i < 7:
                    prev = np.nanmean(row[i:(i+7)])
                    if prev != np.nan:
                        row[i] = prev
                    else :
                        row[i] = 0.0
                elif i > 7:
                    if ( i > 7 ) & (i <= 14):   
                        w = 7
                    elif (i > 14) & (i <= 21):
                        w = 14
                    elif (i > 21) & (i <= 28):
                        w = 21
                    elif (i > 28):
                        w = 28
                    
                    prev = np.nanmean(np.concatenate([row[(i-(w+1)):i:7],row[(i-w):i:7],row[(i-(w-1)):i:7]], axis = -1))

                    if prev != np.nan:
                        row[i] = prev
                    else :
                        row[i] = 0.0
    else :
        row = row.fillna(0)
    return row

class targetSalesScaler(object):
    def __init__(self, lossMask = None):
        self.lossMask = lossMask
        if self.lossMask is not None:
            self.lossMask = lossMask.copy()
            for k in self.lossMask.keys():
                tmp = self.lossMask[k]
                self.lossMask[k] = np.where(tmp == 0, np.nan, tmp)

    def fit_transform(self, data):
        assert isinstance(data, dict)
        keys = data.keys()
        self.index = data[list(keys)[0]].index
        self.columns = data[list(keys)[0]].columns
        length = data[list(keys)[0]].shape[-1]

#         values_ = [v.multiply(l, axis = 0) for v, l in zip(data.values(), self.lossMask.values())]
        values_ =  [v for _, v in data.items()]
        values_ = pd.concat(values_, axis = 1)
        
        
        self.q50 = pd.Series(data = np.nanquantile(values_.values, 0.5, axis = 1), index = values_.index)
        self.q25 = np.nanquantile(values_.values, 0.25, axis = 1)
        self.q75 = np.nanquantile(values_.values, 0.75, axis = 1)
        self.iqr = pd.Series(data = (self.q75 - self.q25), index = values_.index) 
  
        values = [v for _,v in data.items()]
        values = pd.concat(values, axis = 1)
        values = ((values.T - self.q50) / self.iqr).T.fillna(0)
        
        # clipping
        scaled_q25 = values.quantile(0.25, axis = 1)
        scaled_q75 = values.quantile(0.75, axis = 1)
        scaled_iqr = scaled_q75 - scaled_q25
                             
        minimum = scaled_q25 - 1.5 * scaled_iqr    
        maximum = scaled_q75 + 1.5 * scaled_iqr
        values = values.clip(lower = minimum, upper = maximum, axis = 0)
                             
        values = [values.iloc[:,(i*length):((i+1)*length)] for i in range(25)]
        return {k:v for k,v in zip(keys, values)}
        
class StandardScaler(object):
    def __init__(self, lossMask = None):
        super(StandardScaler, self).__init__()
        self.lossMask = lossMask
        if self.lossMask is not None:
            self.lossMask = lossMask.copy()
            for k in self.lossMask.keys():
                tmp = self.lossMask[k] 
                self.lossMask[k] = np.where(tmp == 0, np.nan, tmp)
                
    def fit_transform(self, data):
        if isinstance(data, dict):
            # print("using dict")
           # dataframe in dictionary
            keys = data.keys()
            self.index = data[list(keys)[0]].index
            self.columns = data[list(keys)[0]].columns
            length = data[list(keys)[0]].shape[-1]

            values = [v for _,v in data.items()]
            values = pd.concat(values, axis = 1)
            self.mean = values.mean(axis=1)
            self.std = values.std(axis=1)

            values = ((values.T - self.mean) / self.std).T.fillna(0)
            values = [values.iloc[:,(i*length):((i+1)*length)] for i in range(25)]
            return {k:v for k,v in zip(keys, values)}

        elif isinstance(data, pd.DataFrame):
            # print("using df")
            # values = data.values
            self.mean = data.mean(axis = 1)
            self.std = data.std(axis = 1)
            return ((data.T - self.mean) / self.std).T.fillna(0)
            # return data

        elif isinstance(data, np.ndarray):
            # print("using np")
            self.mean = data.mean()
            self.std = data.std()
            return ((data - self.mean) / self.std)

    def inverse_transform(self, data, target = True):
        """ Not working!!! """
        if target:
            # assert data.shape[1] == config.n, "Input shape should be [None, config.n]"
            step1 = np.prod([data, self.std.values], axis=-1)
            step2 = np.sum([step1, self.mean.values],axis=-1)
            return step2
            # return ((data.T * self.std) + self.mean).T.values
        # assume that data type is numpy or tensor
        else :
            return ((data * self.std) + self.mean)


class Dataloader(object):
    def __init__(self, config):
        super(Dataloader, self).__init__()
        ##########################################################################################
        # Config
        ##########################################################################################
        # self.windowsize = config.k                          # windowsize
        self.c = config.c
        # self.massInfection_threshold = config.massInfection_threshold                           # mass infection threshold
        self.targetPeriod = config.p                        # target period
        
        self.data_startdate = datetime.datetime.strptime(config.data_startdate, "%Y-%m-%d")
        self.target_startdate = config.target_startdate
        self.target_enddate = config.target_enddate
        # self.train_startdate = config.train_startdate
        # self.train_enddate = config.train_enddate
        # self.test_startdate = config.test_startdate
        # self.test_enddate = config.test_enddate

        self.data_dir = config.data_dir
        self.duration = config.duration
        self.sparsity_threshold = config.sparsity_threshold
        self.days = config.days

        assert self.data_startdate < datetime.datetime.strptime(self.target_startdate, "%Y-%m-%d"), "Check config, the start date of input data can not be later than start date of target data"
        
        # enddate - (startdate - windowsize)
        self.maxlen = (datetime.datetime.strptime(self.target_enddate,"%Y-%m-%d") - \
                        self.data_startdate).days - \
                        self.targetPeriod

        # dataframe
        self._majorindustry_amt = pickle.load(open(os.path.join(self.data_dir, "majorIndustry_AMT.pkl"),"rb"))
        self._majorindustry_cnt = pickle.load(open(os.path.join(self.data_dir, "majorIndustry_CNT.pkl"),"rb"))
        self._majorindustry_shop_cnt = pickle.load(open(os.path.join(self.data_dir, "majorIndustry_numOfShopCNT.pkl"),"rb"))
        self._majorindustry_shop_norm = pickle.load(open(os.path.join(self.data_dir, "majorIndustry_numOfShopNORM.pkl"),"rb"))

        # numpy
        fnames = glob.glob(os.path.join(self.data_dir,"*CustDist_{}.pkl".format("AMT")))
        self._custdist_amt = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb")).fillna(0)
            data.sort_index(inplace=True)
            data = data.reindex(sorted(data.columns), axis=1)
            self._custdist_amt[city] = data
        self._custdist_amt = dict(sorted(self._custdist_amt.items()))

        fnames = glob.glob(os.path.join(self.data_dir,"*CustDist_{}.pkl".format("CNT")))
        self._custdist_cnt = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb")).fillna(0)
            data.sort_index(inplace=True)
            data = data.reindex(sorted(data.columns), axis=1)
            self._custdist_cnt[city] = data
        self._custdist_cnt = dict(sorted(self._custdist_cnt.items()))

        self._cdistance = pickle.load(open(os.path.join(self.data_dir, "contextual_distance_matrix.pkl"),"rb"))
        self._pdistance = pickle.load(open(os.path.join(self.data_dir, "physical_distance.pkl"),"rb"))

        # Don't fillna(0)
        self._covid_metainfo = pickle.load(open(os.path.join(self.data_dir,"seoul_mass_infection_metainfo.pkl"),"rb"))
        self._covid_metainfo = self._covid_metainfo.sort_values("Case")
        self._covid_metainfo = self._covid_metainfo.reset_index(drop=True)
        self._covid_metainfo["Startdate"] = pd.to_datetime(self._covid_metainfo["Startdate"])
        self._covid_metainfo["Enddate"] = pd.to_datetime(self._covid_metainfo["Enddate"])
        
        idx = self._covid_metainfo[(self._covid_metainfo.Included == 1) & (self._covid_metainfo.Startdate < self.target_enddate)].index # training set 기준으로 발생한 집단감염 index 찾기
        self._covid_metainfo = self._covid_metainfo.iloc[idx,:]
        self._covid_metainfo = self._covid_metainfo.reset_index(drop=True)

        self._covid_daily = pickle.load(open(os.path.join(self.data_dir,"daily_seoul_mass_infection.pkl"),"rb"))
        self._covid_daily = self._covid_daily.sort_values("Case")
        self._covid_daily = self._covid_daily.reset_index(drop=True)
        
        self._covid_daily = self._covid_daily.iloc[idx,:]
        self._covid_daily = self._covid_daily.reset_index(drop=True)
        # covid_daily = covid_daily[covid_daily.Case == "이태원 클럽 관련"]

        self._covid_cum = pickle.load(open(os.path.join(self.data_dir, "cumulative_seoul_mass_infection.pkl"),"rb"))
        self._covid_cum = self._covid_cum.sort_values("Case")
        self._covid_cum = self._covid_cum.reset_index(drop=True)
        
        self._covid_cum = self._covid_cum.iloc[idx,:]
        self._covid_cum = self._covid_cum.reset_index(drop=True)
        # covid_cum = covid_cum[covid_cum.Case == "이태원 클럽 관련"]

        
        self._covid_re_cum = pickle.load(open(os.path.join(self.data_dir, "recent_cumulative_seoul_mass_infection.pkl"),"rb"))
        self._covid_re_cum = self._covid_re_cum.sort_values("Case")
        self._covid_re_cum = self._covid_re_cum.reset_index(drop=True)
        
        self._covid_re_cum = self._covid_re_cum.iloc[idx,:]
        self._covid_re_cum = self._covid_re_cum.reset_index(drop=True)
        # covid_re_cum = covid_re_cum[covid_re_cum.Case == "이태원 클럽 관련"]


        self._covid_elapsed = pickle.load(open(os.path.join(self.data_dir,"covid_elapsed_day.pkl"),"rb"))
        self._covid_elapsed = self._covid_elapsed.sort_values("Case")
        self._covid_elapsed = self._covid_elapsed.reset_index(drop=True)


        self._covid_elapsed = self._covid_elapsed.iloc[idx,:]
        self._covid_elapsed = self._covid_elapsed.reset_index(drop=True)

        fnames = glob.glob(os.path.join(self.data_dir,"*targetSales_2020_AMT.pkl"))
        self._targetSales = {}
        self._lossMask = {}
        self._targetSales2019 = {}
        self._targetSales2020 = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb"))
            data.sort_index(inplace=True)
            
            # fill 2020-02-29 with 2020-02-22 
            data.loc[:,datetime.datetime.strptime("2020-02-29","%Y-%m-%d")] = data["2020-02-22"]
            data = data.reindex(sorted(data.columns), axis=1)
            
            # 설 연휴 전 주로 대체
            data["2020-01-24"] = data["2020-01-17"]
            data["2020-01-25"] = data["2020-01-18"]
            data["2020-01-26"] = data["2020-01-19"]
            data["2020-01-27"] = data["2020-01-20"] # 대체휴일

            # 추석 연휴 전 주로 대체
            data["2020-09-30"] = data["2020-09-23"]
            data["2020-10-01"] = data["2020-09-24"]
            data["2020-10-02"] = data["2020-09-25"]

            sparsity = data.loc[:,:self.duration].isna().sum(axis=1) / len(data.loc[:,:self.duration])
            interpolation = sparsity < self.sparsity_threshold
            self._lossMask[city] = interpolation.astype(float)
            
            # Interpolation & smoothing
            data = data.apply(lambda x : interpolation_na(x, self.duration, self.sparsity_threshold), axis = 1)
            data = data.fillna(0)
            data = data.ewm(alpha = 0.5, min_periods = 1, axis = 1).mean()
            self._targetSales2020[city] = data

            try :
                os.path.isfile(os.path.join(self.data_dir, "{}_targetSales_2019_AMT.pkl".format(city)))
            except FileExistsError:
                print("Check data_dir in Config!")
                break
            data2019 = pickle.load(open(os.path.join(self.data_dir, "{}_targetSales_2019_AMT.pkl".format(city)),"rb"))
            data2019.sort_index(inplace=True)
            
            # 설연휴 전 주로 대체
            data2019["2019-02-04"] = data2019["2019-01-28"]
            data2019["2019-02-05"] = data2019["2019-01-29"]
            data2019["2019-02-06"] = data2019["2019-01-30"]

            # 추석 연휴 전 주로 대체
            data2019["2019-09-12"] = data2019["2019-09-05"]
            data2019["2019-09-13"] = data2019["2019-09-06"]
            data2019["2019-09-14"] = data2019["2019-09-07"]

            sparsity = data2019.isna().sum(axis=1) / len(data2019)
            interpolation = sparsity < self.sparsity_threshold
            data2019 = data2019.apply(lambda x : interpolation_na(x, "2019-12-31", self.sparsity_threshold), axis = 1)
            data2019 = data2019.fillna(0)
            data2019 = data2019.ewm(alpha = 0.5, min_periods = 1, axis = 1).mean()
            self._targetSales2019[city] = data2019
            data_shift = data2019.shift(periods = -1, axis = "columns")

            self._targetSales[city] = pd.DataFrame(data = ((data.iloc[:,:-2].reset_index(drop=True).values - data_shift.iloc[:,:-1].reset_index(drop=True).values) / data_shift.iloc[:,:-1].reset_index(drop=True).values), index = data.index, columns = data.iloc[:,:-2].reset_index(drop=True).columns).replace([np.inf, -np.inf], np.nan).fillna(0)

        self._lossMask = dict(sorted(self._lossMask.items()))
        self._targetSales = dict(sorted(self._targetSales.items()))

        self._city_dict = {v:k for k,v in enumerate(self._majorindustry_amt.index.values)}
        self._cat_dict = {v:k for k,v in enumerate(self._majorindustry_amt.columns.values)}
    
        ######################################################################################
        # Scaler
        ######################################################################################
        # dataframe
        self.scaler_majorindustry_amt = StandardScaler()
        self._majorindustry_amt_scaled = self.scaler_majorindustry_amt.fit_transform(self._majorindustry_amt) # pd.DataFrame
        # dataframe
        self.scaler_majorindustry_cnt = StandardScaler()
        self._majorindustry_cnt_scaled = self.scaler_majorindustry_cnt.fit_transform(self._majorindustry_cnt) # pd.DataFrame
        # dataframe
        self.scaler_majorindustry_shop_cnt = StandardScaler()
        self._majorindustry_shop_cnt_scaled = self.scaler_majorindustry_shop_cnt.fit_transform(self._majorindustry_shop_cnt) # pd.DataFrame
        # dataframe
        self.scaler_majorindustry_shop_norm = StandardScaler()
        self._majorindustry_shop_norm_scaled = self.scaler_majorindustry_shop_norm.fit_transform(self._majorindustry_shop_norm) # pd.DataFrame
        # dataframe
        self.scaler_custdist_amt = StandardScaler()
        self._custdist_amt_scaled = self.scaler_custdist_amt.fit_transform(self._custdist_amt) # dict
        # dataframe
        self.scaler_custdist_cnt = StandardScaler()
        self._custdist_cnt_scaled = self.scaler_custdist_cnt.fit_transform(self._custdist_cnt) # dict
        # dataframe
        self.scaler_pdistance = StandardScaler()
        self._pdistance_scaled = self.scaler_pdistance.fit_transform(self._pdistance)
        # dataframe
        self.scaler_cdistance = StandardScaler()
        self._cdistance_scaled = self.scaler_cdistance.fit_transform(self._cdistance) #
        
        # dataframe
        # self.scaler_covid_daily = StandardScaler()
        # self._covid_daily_scaled = self._covid_daily.copy()
        # self._covid_daily_scaled.iloc[:,2:] = self.scaler_covid_daily.fit_transform(self._covid_daily_scaled.iloc[:,2:]) # np.array
        self._covid_daily_rolled = self._covid_daily.copy()
        self._covid_daily_rolled.iloc[:,2:] = rolling(self._covid_daily_rolled.iloc[:,2:].values)
        self._covid_daily_scaled_intra = self._covid_daily_rolled.copy()
        self._covid_daily_scaled_intra.iloc[:,2:] = (self._covid_daily_scaled_intra.iloc[:,2:].values.T / self._covid_daily_scaled_intra.iloc[:,2:].values.sum(axis=1)).T
        self._covid_daily_scaled_inter = self._covid_daily_rolled.copy()
        self._covid_daily_scaled_inter.iloc[:,2:] = (self._covid_daily_scaled_inter.iloc[:,2:].values / self._covid_daily_scaled_inter.iloc[:,2:].values.sum(axis = 0))
        self._covid_daily_scaled_inter.iloc[:,2:] = np.nan_to_num(self._covid_daily_scaled_inter.iloc[:,2:].values)

        # # dataframe
        # self.scaler_covid_cum = StandardScaler()
        # self._covid_cum_scaled = self._covid_cum.copy()
        # self._covid_cum_scaled.iloc[:,2:] = self.scaler_covid_cum.fit_transform(self._covid_cum_scaled.iloc[:,2:]) # np.array
        self._covid_cum_rolled = self._covid_cum.copy()
        self._covid_cum_rolled.iloc[:,2:] = rolling(self._covid_cum_rolled.iloc[:,2:].values)
        self._covid_cum_scaled_intra = self._covid_cum_rolled.copy()
        self._covid_cum_scaled_intra.iloc[:,2:] = (self._covid_cum_scaled_intra.iloc[:,2:].T / self._covid_cum_scaled_intra.iloc[:,2:].sum(axis=1)).T
        self._covid_cum_scaled_inter = self._covid_cum_rolled.copy()
        self._covid_cum_scaled_inter.iloc[:,2:] = (self._covid_cum_scaled_inter.iloc[:,2:] / self._covid_cum_scaled_inter.iloc[:,2:].sum(axis=0))
        self._covid_cum_scaled_inter.iloc[:,2:] = np.nan_to_num(self._covid_cum_scaled_inter.iloc[:,2:].values)
        
        # # dataframe
        # self.scaler_covid_re_cum = StandardScaler()
        # self._covid_re_cum_scaled = self._covid_re_cum.copy()
        # self._covid_re_cum_scaled.iloc[:,2:] = self.scaler_covid_re_cum.fit_transform(self._covid_re_cum_scaled.iloc[:,2:]) # np.array
        self._covid_re_cum_rolled = self._covid_re_cum.copy()
        self._covid_re_cum_rolled.iloc[:,2:] = rolling(self._covid_re_cum_rolled.iloc[:,2:].values)
        self._covid_re_cum_scaled_intra = self._covid_re_cum_rolled.copy()
        self._covid_re_cum_scaled_intra.iloc[:,2:] = (self._covid_re_cum_scaled_intra.iloc[:,2:].T / self._covid_re_cum_scaled_intra.iloc[:,2:].sum(axis=1)).T
        self._covid_re_cum_scaled_inter = self._covid_re_cum_rolled.copy()
        self._covid_re_cum_scaled_inter.iloc[:,2:] = (self._covid_re_cum_scaled_inter.iloc[:,2:] / self._covid_re_cum_scaled_inter.iloc[:,2:].sum(axis=0))
        self._covid_re_cum_scaled_inter.iloc[:,2:] = np.nan_to_num(self._covid_re_cum_scaled_inter.iloc[:,2:].values)

        # dataframe
        self.scaler_covid_elapsed = StandardScaler()
        self._covid_elapsed_scaled = self._covid_elapsed.copy()
        self._covid_elapsed_scaled.iloc[:,2:] = self.scaler_covid_elapsed.fit_transform(self._covid_elapsed_scaled.iloc[:,2:]) # np.array

        self.scaler_targetSales = targetSalesScaler(self._lossMask)
        self._targetSales_scaled = self.scaler_targetSales.fit_transform(self._targetSales) # dict

    ######################################################################################
    # Create dataset for target city & mass infection cases at startdate
    # 
    ######################################################################################
    def get_data(self, target, enddate):
 
        startdate_inputs = self.data_startdate
        enddate_inputs = datetime.datetime.strptime(enddate, "%Y-%m-%d")   # enddate of the inputs
 
        # enddate + 1
        startdate_target = enddate_inputs + datetime.timedelta(days = 1) # startdate of targets
        enddate_target = startdate_target + datetime.timedelta(days = self.targetPeriod - 1) # enddate of targets
        
        assert startdate_inputs < startdate_target, "startdate of inputs should be eariler than enddate of targets"
        assert target in self._majorindustry_amt.index , "Out of cities!"
        # startdate is for target range.
        assert startdate_target + datetime.timedelta(days = self.targetPeriod - 1) <= datetime.datetime.strptime(self.target_enddate, "%Y-%m-%d") , "Out of date!"
        
        #################################################
        # List of Variables
        #################################################
        startdate_inputs = startdate_inputs
        enddate_inputs = enddate_inputs
        startdate_target = startdate_target
        enddate_target = enddate_target          
        index_target = None            # city that we are intested in
        index_infected = None          # city of mass infection cases
        index_infected_case = None     # name of mass infection cases
        
        ## Model inputs ##
        majorindustry_target = None    # (1) SI inputs, [33,4] [batch, # of mass infection cases, numOfIndustry, 4]
        majorindustry_infected = None  # (2) SI inputs, [33,4] [batch, # of mass infection cases, numOfIndustry, 4]
        custdist_target = None         # (3) SI inputs, [33,27] [batch, # of mass infection cases, numOfIndustry, 27*] * : # of customer types 
        custdist_infected = None       # (4) SI inputs, [33,27] [batch, # of mass infection cases, numOfIndustry, 27*]
        index_target_idx = None        # (5) SI inputs, [1,1] [batch, # of mass infection cases, 1]
        index_infected_idx = None      # (6) SI inputs, [1,1] [batch, # of mass infection cases, 1]
        physical_distance = None       # (7) SA inputs, [1,1] [batch, # of mass infection cases, 1]
        contextual_distance = None     # (8) SA inputs, [1,1] [batch, # of mass infection cases, 1]
        covid_industry = None          # (9) SO inputs, [1,1] [batch, # of mass infection cases, 1]
        severity = None                # (10) S inputs,  [140,3] [batch, # of mass infection cases,  None* - 1, 3**] * : dynamically changed, ** : daily, cumulative, recent_cumulative
        covid_elapsed_day = None       # (11) Elapsed inputs, [141,1] [batch, # of mass infection cases, None*, 1] * : dynamically changed
        weekdays = None                # (14) Weekday inputs, [7,7] [batch, # of mass infection cases, targetPeriod, 7*] * : dummy variables
        mask = None                    # (15) mask for severity, covid_elapsed_day, [134,1] [batch, None*, 1] * : dynamically changed
        covidMask = None               # (16) mask for covid mass infection cases, [1, ] [batch, # of mass infection cases, 1]
        
        targetSale = None              # outputs, [batch, config.n, targetPeriod]
        lossMask = None                # outputs, [batch, config.n]
        # severity_target = None         # outputs, [batch, # of mass infection cases, targetPeriod, 3]
        ####################################################

        index_target = target

        ######################################################
        # Specified Industry Dataset (1)
        ######################################################
        # dataframe
        majorindustry_target = np.stack([
                                        self._majorindustry_amt_scaled.loc[index_target].values,
                                        self._majorindustry_cnt_scaled.loc[index_target].values,
                                        self._majorindustry_shop_cnt_scaled.loc[index_target].values,
                                        self._majorindustry_shop_norm_scaled.loc[index_target].values,
                                        ], axis = -1)
        
        ######################################################
        # Specified Industry Dataset (2)
        ######################################################
        # dictionary
        custdist_target = np.stack([
                                    self._custdist_amt_scaled[index_target].values,
                                    self._custdist_cnt_scaled[index_target].values
                                    ], axis = -1)
        
        majorindustry_infected = []
        case = []
        custdist_infected = []
        physical_distance = []
        contextual_distance = []
        severity = []
        elapsed_day = []

        index_infected = self._covid_daily[["City"]]
        index_infected_case = self._covid_daily[["Case"]]
        covid_industry = self._covid_metainfo["TP_GRP_NM"].map(self._cat_dict)[...,np.newaxis]

        ######################################################
        # Severity Dataset (1) - (3)
        ######################################################
        # encoder data for severity 
        severity_daily = np.stack([self._covid_daily_scaled_intra.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values,
                                    self._covid_daily_scaled_inter.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values], axis = -1)
        # severity_mask = np.array(((enddate_inputs + datetime.timedelta(days = -self.days) < self._covid_metainfo.Startdate) & (self._covid_metainfo.Startdate < enddate_inputs)) |\
        #     ((enddate_inputs + datetime.timedelta(days = -self.days) < self._covid_metainfo.Enddate) & (self._covid_metainfo.Enddate < enddate_inputs)) |\
        #         ((enddate_inputs + datetime.timedelta(days = -self.days) > self._covid_metainfo.Startdate) & (enddate_inputs > self._covid_metainfo.Enddate))).astype(float)

        # severity_daily_q50, severity_daily_iqr = epidemicScaler(severity_daily[np.nonzero(severity_mask),-self.days:])

        # severity_daily_scaled = (severity_daily - severity_daily_q50) / severity_daily_iqr
        # print(severity_daily_q50, severity_daily_iqr, severity_daily_scaled[:,-self.days:], end="")

        severity_daily = tf.keras.preprocessing.sequence.pad_sequences(severity_daily, 
                                                                                    maxlen=self.maxlen, 
                                                                                    dtype='float32',
                                                                                    padding = 'post')

        # sevirity_daily_mask = tf.sequence_mask(severity_daily.shape[0], maxlen=self.maxlen)
        severity_cum = np.stack([self._covid_cum_scaled_intra.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values,
                                    self._covid_cum_scaled_inter.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values], axis = -1)
        severity_cum = tf.keras.preprocessing.sequence.pad_sequences(severity_cum, 
                                                                                maxlen=self.maxlen, 
                                                                                dtype='float32',
                                                                                padding = 'post')
        # severity_cum_mask = tf.sequence_mask(severity_cum.shape[0], maxlen=self.maxlen)
        
        severity_re_cum = np.stack([self._covid_re_cum_scaled_intra.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values,
                                    self._covid_re_cum_scaled_inter.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values], axis = -1)
        severity_re_cum = tf.keras.preprocessing.sequence.pad_sequences(severity_re_cum, 
                                                                                    maxlen=self.maxlen, 
                                                                                    dtype='float32',
                                                                                    padding = 'post')
        # severity_re_cum_mask = tf.sequence_mask(severity_re_cum.shape[0], maxlen=self.maxlen)
        
        severity_daily  = np.array(severity_daily)
        severity_cum    = np.array(severity_cum)
        severity_re_cum = np.array(severity_re_cum)
        
        
        # [config.c, 134, 3]
        severity = np.concatenate([severity_daily, severity_cum, severity_re_cum],axis=-1) # [None, n, windowsize, 3]


        # decoder data for severity : [config.c, 7, 3]        
        severity_daily = np.stack([self._covid_daily_scaled_intra.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values,
                                    self._covid_daily_scaled_inter.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values], axis = -1)
       
        severity_cum = np.stack([self._covid_cum_scaled_intra.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values,
                                    self._covid_cum_scaled_inter.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values], axis = -1)

        severity_re_cum = np.stack([self._covid_re_cum_scaled_intra.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values,
                                    self._covid_re_cum_scaled_inter.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values], axis = -1)
        severity_decoder = np.concatenate([severity_daily, severity_cum, severity_re_cum],axis=-1)

        # concatenate encoder + decoder data [config.c, 141, 3]
        severity = np.concatenate([severity, severity_decoder[:,:-1,:]], axis = 1) # [None, 134, 6] + [None, 6, 6]

        ######################################################
        # Elapsed Day Dataset (1)
        ######################################################        
        covid_elapsed_day = self._covid_elapsed_scaled.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        covid_elapsed_day_origin = self._covid_elapsed.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        covid_elapsed_day_target = self._covid_elapsed_scaled.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        # print(covid_elapsed_day_target)
        idx = np.where(covid_elapsed_day_origin[:,-1] == 0)
        covid_elapsed_day_target[idx[0], : ] = tf.tile(tf.expand_dims(covid_elapsed_day[idx[0], -1], axis = -1), [1, covid_elapsed_day_target.shape[-1]])

        covid_elapsed_day = tf.keras.preprocessing.sequence.pad_sequences(covid_elapsed_day, 
                                                                        maxlen=self.maxlen, 
                                                                        dtype='float32',
                                                                        padding = 'post')
        covid_elapsed_day = np.concatenate([covid_elapsed_day, covid_elapsed_day_target], axis = 1)

        ######################################################
        # Specified Industry Dataset (3),(4) / SimInArea Dataset (1),(2)
        ######################################################   
        # generate dataset for all mass infection cases
        for i in range(index_infected.shape[0]):
            city = index_infected.loc[i, "City"]
            majorindustry_infected.append(np.stack([self._majorindustry_amt_scaled.loc[city].values,
                                                    self._majorindustry_cnt_scaled.loc[city].values,
                                                    self._majorindustry_shop_cnt_scaled.loc[city].values,
                                                    self._majorindustry_shop_norm_scaled.loc[city].values,
                                                    ], axis = -1)
            )
            custdist_infected.append(np.stack([self._custdist_amt_scaled[city].values,
                                               self._custdist_cnt_scaled[city].values], axis = -1)) # dictionary
            physical_distance.append([[self._pdistance_scaled.loc[city, index_target]]])
            contextual_distance.append([[self._cdistance_scaled.loc[city, index_target]]])

        majorindustry_infected = np.array(majorindustry_infected) # [None, n, 35]
        custdist_infected = np.array(custdist_infected) # [None, n, 35, 27]

        physical_distance = np.array(physical_distance)
        contextual_distance = np.array(contextual_distance) # [None, n, 1]

        seqlen = (enddate_inputs - startdate_inputs).days + 1
        mask = tf.sequence_mask(seqlen, maxlen=self.maxlen, dtype = tf.dtypes.float32).numpy()

        
        #################################################
        # Target Dataset
        #################################################
        # target data
        targetSales = self._targetSales_scaled[index_target].loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        # Loss Mask
        lossMask = self._lossMask[index_target]
        # Mass Infection Mask
        covidMask = np.array((self._covid_metainfo.Startdate < enddate_inputs) & (self._covid_metainfo.Enddate + datetime.timedelta(days = self.days) > enddate_inputs)).astype(float)
        # idx = np.where(covidMask == 0 )
        # covidMask[idx] = -float('inf')

        # weekdays = np.zeros(shape=(self.targetPeriod, 7))
        weekdays = [(startdate_target + datetime.timedelta(days=x)).weekday() for x in range(self.targetPeriod)] # shape = [7]
        weekdays = np.array(weekdays)
        # for i in range(len(weekdays_)):
        #     weekdays[i,weekdays_[i]] = 1

        index_target_idx = np.array([self._city_dict[index_target]])
        index_infected_idx = np.array([[self._city_dict[idx]] for idx in index_infected["City"]])
        #################################################
        # Create dataset 
        #################################################
        numOfCases = self.c
        dataset = []
        for i in range(numOfCases) : 
            dataset.append(
                np.array([
                    majorindustry_target, \
                    majorindustry_infected[i,:], \
                    custdist_target, \
                    custdist_infected[i,:], \
                    index_target_idx[...,np.newaxis], \
                    index_infected_idx[i][...,np.newaxis], \
                    physical_distance[i,:], \
                    contextual_distance[i,:], \
                    covid_industry[i,:][...,np.newaxis], \
                    severity[i,:],
                    covid_elapsed_day[i,:][..., np.newaxis],
                    weekdays[..., np.newaxis],
                    mask[...,np.newaxis],
                    covidMask[i][...,np.newaxis]
                ])
                    # targetSales
            )        
        metadata = np.array([index_target, startdate_target, enddate_target])
        return dataset, targetSales, lossMask, metadata

    ###################################################################################################
    # Create train dataset for all target cities & mass infection cases from 2020.02.01 ~ 2020.06.16
    # 
    ###################################################################################################
    @property
    def datasets(self):
        startdate = datetime.datetime.strptime(self.target_startdate, "%Y-%m-%d") - \
            datetime.timedelta(days = 1)
        enddate = datetime.datetime.strptime(self.target_enddate, "%Y-%m-%d") - \
            datetime.timedelta(days = self.targetPeriod - 1) # range in for loop

        target_cities = self._majorindustry_amt.index

        inputs = []
        labels = []
        lossMasks = []
        labels_severity = []
        metadatas = []
        for ed in self.daterange(startdate, enddate):
            ed = ed.strftime("%Y-%m-%d") # enddate of inputs
            print("\rProcessing *Input* Dataset between {} ~ {}".format(self.data_startdate,ed), end="")
            for city in target_cities:
                dataset, targetSales, lossMask, metadata = self.get_data(city, ed)
                # if targetSales.shape[1] != 7:
                #     print(targetSales.shape)
                inputs.append(dataset)
                labels.append(targetSales)
                lossMasks.append(lossMask)
                metadatas.append(metadata)
        # inputs : [day*city, # of mass infection cases, 12, None, None]
        return np.array(inputs), np.array(labels), np.array(lossMasks), \
           np.array(metadatas)

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + datetime.timedelta(n)
#%%
#######################################################################
# Training Utilities
#######################################################################
def create_month_data(config, datasets, start = 2, end = 10, testdays = 14, interval = 1, phase = 1):
    '''
    # https://stackoverflow.com/questions/13648774/get-year-month-or-day-from-numpy-datetime64
    # https://stackoverflow.com/questions/42950/how-to-get-the-last-day-of-the-month
    datasets : datasets from dataloader which contains X, y, lossMask, severity, metadata
    start    : set the first month that we want to train
    end      : set the last month that we want to train
    testdays : the duration of testset, default = 14
    interval : interval between months, default = 1
    phase    : targetPeriod = 7 -> phase : 1 ~ 4
    '''
    assert start >= 2, "start should be larger than 2"
    print("start month: {}, end month: {}, testdays: {}, interval: {}".format(start, end, testdays, interval))
    X, y, lossMask, Severity, metadata = datasets

    startdate = np.array([datetime.datetime(2020, start, 1) + relativedelta(months = i) for i in range(0, end - start + 1, interval)])
    startdate = np.insert(startdate, 0,  datetime.datetime(2020,1,1))
    testdate = np.array([st + datetime.timedelta(days = testdays) for st in startdate])
    _next_months = np.array([st.replace(day=28) + datetime.timedelta(days=4) for st in startdate])
    enddate = np.array([nm - datetime.timedelta(nm.day) for nm in _next_months])
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    startdate_target = metadata[:,1]
    enddate_target = metadata[:,2]
    train_datasets = []
    test_datasets = []
    for i in range(1, len(enddate)):
        if i == len(enddate) - 1 :
            revised_enddate_target_train = startdate[i] + datetime.timedelta(days = (phase-1)*7) # 2020-10-01 + 0/7/14/21
            print("-Training Set   {}: {} ~ {}".format(i, startdate[i], revised_enddate_target_train))
            m1 = startdate_target >= startdate[i]
            m2 = enddate_target < revised_enddate_target_train # 2020-10-01 + 0/7/14/21
            idx = m1 & m2
            if (idx.sum() == 0):
                continue
            X_train = X[idx, :, :]
            train_dataset = (X_train, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            train_datasets.append(train_dataset)

            revised_startdate_target_test = startdate[i] + datetime.timedelta(days = (phase-1)*7) # 2020-10-01 + 0/7/14/21
            print("-Validating Set {}: {} ~ {}".format(i, revised_startdate_target_test, enddate[i]))
            m3 = startdate_target >= revised_startdate_target_test # 2020-10-01 + 0/7/14/21
            m4 = enddate_target <= enddate[i]
            idx = m3 & m4
            X_test = X[idx, :, :]
            X_test[:,:,-1] = X_train[:,:,-1][-1] # train에서는 등장하지 않았으나 test에서 등장한 집단감염은 masking을 0으로 바꿔줘야 함.
            test_dataset = (X_test, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            test_datasets.append(test_dataset)

        else :
            print("Training Set   {}: {} ~ {}".format(i, startdate[i], startdate[i+1]))
            m1 = startdate_target >= startdate[i]
            m2 = enddate_target < startdate[i+1]
            idx = m1 & m2
            X_train = X[idx, :, :]
            train_dataset = (X_train, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            train_datasets.append(train_dataset)

            print("Validating Set {}: {} ~ {}".format(i, startdate[i+1], testdate[i+1]))
            m3 = startdate_target >= startdate[i+1]
            m4 = enddate_target < testdate[i+1]
            idx = m3 & m4
            X_test = X[idx, :, :]
            X_test[:,:,-1] = X_train[:,:,-1][-1] # train에서는 등장하지 않았으나 test에서 등장한 집단감염은 masking을 0으로 바꿔줘야 함.
            test_dataset = (X_test, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            test_datasets.append(test_dataset)
    print("train_datasets {}, test_datasets {}".format(len(train_datasets),len(test_datasets)))
    return train_datasets, test_datasets

def loadData(config, dataloader, existing = True) :
    if os.path.isfile(config.datapath) & existing :
        print("File Exists at {}".format(config.datapath))
        datasets = pickle.load(open(config.datapath, "rb"))
    else :
        print("File not Exists at {}".format(config.datapath))
        datasets = dataloader.datasets # X_train : [None, 8, 12, ...]
        pickle.dump(datasets, \
                    open(config.datapath, "wb"))
    print("X shape {}, y shape {}, lossMask shape {} Severity shape {}".format(datasets[0].shape, datasets[1].shape, datasets[2].shape, datasets[3].shape))
    return datasets

def create_month_data_v2(config, datasets, start = 2, end = 12, testdays = 14, interval = 1, phase = 1):
    '''
    # https://stackoverflow.com/questions/13648774/get-year-month-or-day-from-numpy-datetime64
    # https://stackoverflow.com/questions/42950/how-to-get-the-last-day-of-the-month
    datasets : datasets from dataloader which contains X, y, lossMask, severity, metadata
    start    : set the first month that we want to train
    end      : set the last month that we want to train
    testdays : the duration of testset, default = 14
    interval : interval between months, default = 1
    phase    : targetPeriod = 7 -> phase : 1 ~ 4
    '''
    assert start >= 2, "start should be larger than 2"
    print("start month: {}, end month: {}, testdays: {}, interval: {}".format(start, end, testdays, interval))
    X, y, lossMask, Severity, metadata = datasets

    startdate = np.array([datetime.datetime(2020, start, 1) + relativedelta(months = i) for i in range(0, end - start + 1, interval)])
    startdate = np.insert(startdate, 0,  datetime.datetime(2020,1,1))
    testdate = np.array([st + datetime.timedelta(days = testdays) for st in startdate])
    _next_months = np.array([st.replace(day=28) + datetime.timedelta(days=4) for st in startdate])
    enddate = np.array([nm - datetime.timedelta(nm.day) for nm in _next_months])
    # subtract the number of remaining 'overage' days to get last day of current month, or said programattically said, the previous day of the first of next month
    startdate_target = metadata[:,1]
    enddate_target = metadata[:,2]
    train_datasets = []
    test_datasets = []
    for i in range(1, len(enddate)):
        if i == len(enddate) - 1 :
            revised_enddate_target_train = startdate[i] + datetime.timedelta(days = (phase-1)*7) # 2020-10-01 + 0/7/14/21
            print("Training Set   {}: Y STARTDATE is between {} and {}".format(i-1, startdate[i] + datetime.timedelta(days = 1), startdate[i] + datetime.timedelta(days = 15 - config.p)))
            m1 = startdate_target >= startdate[i] + datetime.timedelta(days = 1) # x ends at 2020-12-01 & y starts from 2020-12-01 + 1 => 2020-12-02
            if config.p == 7 :
                m2 = startdate_target <= startdate[i] + datetime.timedelta(days = 8) # x ends at 2020-12-08 & y starts from 2020-12-01 + 8 => 2020-12-09 to 2020-12-15
            elif config.p > 7 :
                m2 = startdate_target <= startdate[i] + datetime.timedelta(days = 1) # x ends at 2020-12-08 & y starts from 2020-12-01 + 8 => 2020-12-09 to 2020-12-15
            idx = m1 & m2
            if (idx.sum() == 0):
                continue
            X_train = X[idx, :, :]
            train_dataset = (X_train, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            train_datasets.append(train_dataset)

            print("Validating Set {}: Y STARTDATE is between {} and {}".format(i-1, startdate[i] + datetime.timedelta(days = 15), startdate[i] + datetime.timedelta(days = 15 + 14 - config.p)))
            m3 = startdate_target >= startdate[i] + datetime.timedelta(days = 15) # x ends at 2020-12-15 & y starts from 2020-12-01 + 15 => 2020-12-16
            if config.p == 7 :
                m4 = startdate_target <= startdate[i] + datetime.timedelta(days = 15 + 7) # x ends at 2020-12-22 & y starts from 2020-12-01 + 22 => 2020-12-23
            elif config.p > 7 :
                m4 = startdate_target <= startdate[i] + datetime.timedelta(days = 15) # x ends at 2020-12-22 & y starts from 2020-12-01 + 22 => 2020-12-23
            idx = m3 & m4
            X_test = X[idx, :, :]
            X_test[:,:,-1] = X_train[:,:,-1][-1] # train에서는 등장하지 않았으나 test에서 등장한 집단감염은 masking을 0으로 바꿔줘야 함.
            test_dataset = (X_test, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            test_datasets.append(test_dataset)

        else :
            print("Training Set   {}: Y STARTDATE is between {} and {}".format(i-1, startdate[i] + datetime.timedelta(days = 1), startdate[i+1]))
            m1 = startdate_target >= startdate[i] + datetime.timedelta(days = 1) # x ends at 2020-03-01 & y starts from 2020-03-01 + 1 => 2020-03-02
            m2 = startdate_target <= startdate[i+1] # x ends at 2020-03-31 & y starts from 2020-04-01
            idx = m1 & m2
            X_train = X[idx, :, :]
            train_dataset = (X_train, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            train_datasets.append(train_dataset)

            # if config.p == 7:
            print("Validating Set {}: Y STARTDATE is between {} and {}\n".format(i-1, startdate[i+1] + datetime.timedelta(days = 1), startdate[i+1] + datetime.timedelta(days = 15 - config.p)))
            m3 = startdate_target >= startdate[i+1] + datetime.timedelta(days = 1) # x ends at 2020-04-01 & y starts from 2020-04-01 + 1 => 2020-04-02
            m4 = startdate_target <= startdate[i+1] + datetime.timedelta(days = 15 - config.p) # x ends at 2020-03-31 & y starts from 2020-04-01
            idx = m3 & m4
            # elif config.p == 14:
            #     print("Validating Set {}: y *startdate* is from {} to {}".format(i, startdate[i+1] + datetime.timedelta(days = 1), testdate[i+1] + datetime.timedelta(days = config.p)))
            #     m3 = startdate_target >= startdate[i+1] + datetime.timedelta(days = 1) # x ends at 2020-04-01 & y starts from 2020-04-01 + 1 => 2020-04-02
            #     m4 = startdate_target <= startdate[i+1] + datetime.timedelta(days = 15 - config.p) # x ends at 2020-03-31 & y starts from 2020-04-01
            #     idx = m3 & m4
            X_test = X[idx, :, :]
            X_test[:,:,-1] = X_train[:,:,-1][-1] # train에서는 등장하지 않았으나 test에서 등장한 집단감염은 masking을 0으로 바꿔줘야 함.
            test_dataset = (X_test, y[idx,:], lossMask[idx,:], Severity[idx,:,:], metadata[idx, :])
            test_datasets.append(test_dataset)
    print("train_datasets {}, test_datasets {}".format(len(train_datasets),len(test_datasets)))
    return train_datasets, test_datasets

def split_data(config, datasets, test_month):
    X, y, lossMask, metadata = datasets
    daysInMonths = [31,29,31,30,31,30,31,31,30,31,30,29]
    targetPeriod = y.shape[-1]
    daysInMonth = sum(daysInMonths[test_month - 1:]) + targetPeriod
    validdays = sum(daysInMonths[test_month - 1:]) + 1
    testdays = sum(daysInMonths[test_month - 1:]) - targetPeriod + 1
    # train_datasets = (X[:-(daysInMonth*25)], y[:-(daysInMonth*25)], lossMask[:-(daysInMonth*25)], Severity[:-(daysInMonth*25)], metadata[:-(daysInMonth*25)])
    train_datasets = (X[:-(daysInMonth*25)], y[:-(daysInMonth*25)], lossMask[:-(daysInMonth*25)], metadata[:-(daysInMonth*25)])
    valid_datasets = (X[-(validdays*25):-((validdays-1)*25)], y[-(validdays*25):-((validdays-1)*25)], lossMask[-(validdays*25):-((validdays-1)*25)], metadata[-(validdays*25):-((validdays-1)*25)])
    covidMask = train_datasets[0][-1,:,-1]
    if y.shape[-1] == 7:
        X_test = np.concatenate([X[-(testdays*25):][:25], X[-(testdays*25):][25*7:25*8]], axis = 0)
        X_test[:,:,-1] = covidMask
        test_datasets = (
                        X_test,
                        np.concatenate([y[-(testdays*25):][:25], y[-(testdays*25):][25*7:25*8]], axis = 0),
                        np.concatenate([lossMask[-(testdays*25):][:25], lossMask[-(testdays*25):][25*7:25*8]], axis = 0),
                        np.concatenate([metadata[-(testdays*25):][:25], metadata[-(testdays*25):][25*7:25*8]], axis = 0)
                        )
    elif y.shape[-1] >= 14:
        X_test = X[-(testdays*25):][:25]
        X_test[:,:,-1] = covidMask
        test_datasets = (
                        X_test,
                        y[-(testdays*25):][:25],
                        lossMask[-(testdays*25):][:25],
                        metadata[-(testdays*25):][:25]
                        )
    print("Train data startdate: {} ~ {}".format(train_datasets[-1][0][1], train_datasets[-1][-1][1]))
    print("Valid data startdate: {} ~ {}".format(valid_datasets[-1][0][1], valid_datasets[-1][-1][1]))
    print("Test data startdate : {} ~ {}".format(test_datasets[-1][0][1], test_datasets[-1][-1][1]))
    
    return train_datasets, valid_datasets, test_datasets
    
def loadData(config, dataloader, existing = True) :
    if os.path.isfile(config.datapath) & existing :
        print("File Exists at {}".format(config.datapath))
        datasets = pickle.load(open(config.datapath, "rb"))
    else :
        print("File not Exists at {}".format(config.datapath))
        datasets = dataloader.datasets # X_train : [None, 8, 12, ...]
        pickle.dump(datasets, \
                    open(config.datapath, "wb"))
    print("X shape {}, y shape {}, lossMask shape {} Severity shape {}".format(datasets[0].shape, datasets[1].shape, datasets[2].shape, datasets[3].shape))
    return datasets
# %%
def draw_figure(config, epoch, y_true, y_pred, lossMask, metainfo, name = None):
    length = metainfo.shape[0]
    for i, city in enumerate(metainfo[:,0]):
        print("Drawing Figure for {}...".format(city), end="")
        fig, axs = plt.subplots(7, 5, figsize = (20,20), sharex = "all")
        fig.suptitle("{} - {} ~ {} 매출 변화량".format(city, metainfo[i,1].strftime("%Y-%m-%d"), metainfo[i,2].strftime("%Y-%m-%d")))
        y_true_city = y_true[i::length, :, :]    
        y_pred_city = y_pred[i::length, :, :]
        lossMask_city = lossMask[i::length, :]
        
        if len(y_true_city.shape) == 3:
            y_true_city = y_true_city[0,:,:]
            y_pred_city = y_pred_city[0,:,:]
            lossMask_city = lossMask_city[0,:]
        x, y = 0, 0
        buz_dict = config.buz_dict
        for buz in config.buz_dict.keys():
            idx = buz_dict[buz]
            Not_Masked = lossMask_city[idx]
            axs[x, y].plot(y_true_city[idx,:],c="b", label = "y_true")
            axs[x, y].plot(y_pred_city[idx,:],c="r", label = "y_true")
            rmse = (((y_true_city[idx,:] - y_pred_city[idx,:])**2).mean())**0.5
            axs[x, y].set_title("{} - {} - rmse: {:03f}".format(str(Not_Masked),buz, rmse))
            y += 1
            if y >= 5:
                y = 0
                x += 1
        fname = "_{}_{}_test.png".format(city, str(epoch)) if name is None else "_{}_{}_{}_test.png".format(city, str(epoch), name)
        fname = os.path.join(config.checkpoint_figurespath, config.checkpoint_figures) + fname
        fig.tight_layout()
        fig.savefig(fname)
#         plt.show()
#%%
if __name__ == '__main__':
    config = Config(
        data_dir = "../data/preprocess12_new",
        s = 0,
        p = 7,
        c = 30,
        target_enddate = "2020-12-29",
        data_startdate = "2020-11-01",
        target_startdate = "2020-12-20"
    )
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[config.gpu_num], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[config.gpu_num], True)
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
    dataloader = Dataloader(config)
    x,y,z,w,v = dataloader.datasets
    for i in range(x.shape[-1]):
        print(x[0][0][i].shape)# %%
