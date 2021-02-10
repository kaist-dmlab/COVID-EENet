##########################################################################
# Title  : Utilities
# Date   : 2020/12/20
# Update : 2021/02/08
# Author : Hyangsuk
##########################################################################
#%%
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle, glob, os
import warnings
warnings.filterwarnings(action='ignore')

def interpolation_na(row, duration, threshold):
    sparsity = row[:duration].isna().sum() / len(row[:duration])
    interpolation = sparsity < threshold
    if interpolation :
        # print("interpolation {}, sparsity {:3f}%".format(row.name, sparsity))
        isna = row.isna()
        for i in range(len(isna)):
            if isna[i]: # numpy.float64
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

class SalesChangeScaler(object):
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
    

class Scaler(object):
    def __init__(self, lossMask = None):
        super(Scaler, self).__init__()
        self.lossMask = lossMask
        if self.lossMask is not None:
            self.lossMask = lossMask.copy()
            for k in self.lossMask.keys():
                tmp = self.lossMask[k] 
                self.lossMask[k] = np.where(tmp == 0, np.nan, tmp)
                
    def fit_transform(self, data):
        if isinstance(data, dict):
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
            self.mean = data.mean(axis = 1)
            self.std = data.std(axis = 1)
            return ((data.T - self.mean) / self.std).T.fillna(0)

        elif isinstance(data, np.ndarray):
            self.mean = data.mean()
            self.std = data.std()
            return ((data - self.mean) / self.std)

class Dataloader(object):
    def __init__(self, config):
        super(Dataloader, self).__init__()
        ##########################################################################################
        # Config
        ##########################################################################################
        self.m = config.m
        self.targetPeriod = config.w                        # target period

        self.data_dir = config.data_dir
        self.fname_BusinessStructure_amt = config.fname_BusinessStructure_amt
        self.fname_BusinessStructure_cnt = config.fname_BusinessStructure_cnt
        self.fname_CustomerStructure = config.fname_CustomerStructure
        self.fname_contextual_distance = config.fname_contextual_distance
        self.fname_physical_distance = config.fname_physical_distance
        self.fname_covid_metainfo = config.fname_covid_metainfo
        self.fname_covid_daily = config.fname_covid_daily
        self.fname_covid_cum = config.fname_covid_cum
        self.fname_covid_re_cum = config.fname_covid_re_cum
        self.fname_elapsed_day = config.fname_elapsed_day
        self.fname_Sales2020 = config.fname_Sales2020
        self.fname_Sales2019 = config.fname_Sales2019

        self.data_startdate = datetime.datetime.strptime("2020-01-01", "%Y-%m-%d")
        self.target_startdate = datetime.datetime.strptime("2020-02-01", "%Y-%m-%d")
        self.target_enddate = datetime.datetime.strptime("2020-12-29", "%Y-%m-%d")

        assert self.data_startdate < self.target_startdate, "Check config, the start date of input data can not be later than start date of target data"
        # enddate - (startdate - windowsize)
        self.maxlen = (self.target_enddate - \
                        self.data_startdate).days - \
                        self.targetPeriod

        # dataframe
        self._BusinessStructure_amt = pickle.load(open(os.path.join(self.data_dir, self.fname_BusinessStructure_amt),"rb"))
        self._BusinessStructure_cnt = pickle.load(open(os.path.join(self.data_dir, self.fname_BusinessStructure_cnt),"rb"))

        # numpy
        # self._CustomerStructure = pickle.load(open(os.path.join(self.data_dir, "majorIndustry_cust_{}.pkl".format(self.sales)),"rb"))
        fnames = glob.glob(os.path.join(self.data_dir, self.fname_CustomerStructure.format("AMT")))
        self._CustomerStructure_amt = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb")).fillna(0)
            data.sort_index(inplace=True)
            data = data.reindex(sorted(data.columns), axis=1)
            self._CustomerStructure_amt[city] = data
        self._CustomerStructure_amt = dict(sorted(self._CustomerStructure_amt.items()))

        fnames = glob.glob(os.path.join(self.data_dir, self.fname_CustomerStructure.format("CNT")))
        self._CustomerStructure_cnt = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb")).fillna(0)
            data.sort_index(inplace=True)
            data = data.reindex(sorted(data.columns), axis=1)
            self._CustomerStructure_cnt[city] = data
        self._CustomerStructure_cnt = dict(sorted(self._CustomerStructure_cnt.items()))

        self._cdistance = pickle.load(open(os.path.join(self.data_dir, self.fname_contextual_distance),"rb"))
        self._pdistance = pickle.load(open(os.path.join(self.data_dir, self.fname_physical_distance),"rb"))

        # Don't fillna(0)
        self._covid_metainfo = pickle.load(open(os.path.join(self.data_dir, self.fname_covid_metainfo),"rb"))
        self._covid_metainfo = self._covid_metainfo.sort_values("Case")
        self._covid_metainfo = self._covid_metainfo.reset_index(drop=True)
        self._covid_metainfo["Startdate"] = pd.to_datetime(self._covid_metainfo["Startdate"])
        self._covid_metainfo["Enddate"] = pd.to_datetime(self._covid_metainfo["Enddate"])
        
        idx = self._covid_metainfo[(self._covid_metainfo.Included == 1) & (self._covid_metainfo.Startdate < self.target_enddate)].index # training set 기준으로 발생한 집단감염 index 찾기
        self._covid_metainfo = self._covid_metainfo.iloc[idx,:]
        self._covid_metainfo = self._covid_metainfo.reset_index(drop=True)

        self._covid_daily = pickle.load(open(os.path.join(self.data_dir, self.fname_covid_daily),"rb"))
        self._covid_daily = self._covid_daily.sort_values("Case")
        self._covid_daily = self._covid_daily.reset_index(drop=True)
        
        self._covid_daily = self._covid_daily.iloc[idx,:]
        self._covid_daily = self._covid_daily.reset_index(drop=True)

        self._covid_cum = pickle.load(open(os.path.join(self.data_dir, self.fname_covid_cum),"rb"))
        self._covid_cum = self._covid_cum.sort_values("Case")
        self._covid_cum = self._covid_cum.reset_index(drop=True)
        
        self._covid_cum = self._covid_cum.iloc[idx,:]
        self._covid_cum = self._covid_cum.reset_index(drop=True)
        
        self._covid_re_cum = pickle.load(open(os.path.join(self.data_dir, self.fname_covid_re_cum),"rb"))
        self._covid_re_cum = self._covid_re_cum.sort_values("Case")
        self._covid_re_cum = self._covid_re_cum.reset_index(drop=True)
        
        self._covid_re_cum = self._covid_re_cum.iloc[idx,:]
        self._covid_re_cum = self._covid_re_cum.reset_index(drop=True)

        self._covid_elapsed = pickle.load(open(os.path.join(self.data_dir, self.fname_elapsed_day),"rb"))
        self._covid_elapsed = self._covid_elapsed.sort_values("Case")
        self._covid_elapsed = self._covid_elapsed.reset_index(drop=True)
        
        self._covid_elapsed = self._covid_elapsed.iloc[idx,:]
        self._covid_elapsed = self._covid_elapsed.reset_index(drop=True)

        fnames = glob.glob(os.path.join(self.data_dir, self.fname_Sales2020))
        self._SalesChange = {}
        self._lossMask = {}
        self._Sales2019 = {}
        self._Sales2020 = {}
        for fname in fnames:
            city = fname.split("/")[-1].split("\\")[-1].split("_")[0]
            data = pickle.load(open(fname,"rb"))
            data.sort_index(inplace=True)
            
            # fill 2020-02-29 with 2020-02-22 
            data.loc[:,datetime.datetime.strptime("2020-02-29","%Y-%m-%d")] = data["2020-02-22"]
            data = data.reindex(sorted(data.columns), axis=1)
            
            # Holiday
            data["2020-01-24"] = data["2020-01-17"]
            data["2020-01-25"] = data["2020-01-18"]
            data["2020-01-26"] = data["2020-01-19"]
            data["2020-01-27"] = data["2020-01-20"] 

            # Holiday
            data["2020-09-30"] = data["2020-09-23"]
            data["2020-10-01"] = data["2020-09-24"]
            data["2020-10-02"] = data["2020-09-25"]

            sparsity = data.loc[:,:"2020-04-09"].isna().sum(axis=1) / len(data.loc[:,:"2020-04-09"])
            interpolation = sparsity < 0.7
            self._lossMask[city] = interpolation.astype(float)
            
            # Interpolation & smoothing
            data = data.apply(lambda x : interpolation_na(x, "2020-04-09", 0.7), axis = 1)
            data = data.fillna(0)
            data = data.ewm(alpha = 0.5, min_periods = 1, axis = 1).mean()
            self._Sales2020[city] = data

            try :
                os.path.isfile(os.path.join(self.data_dir, self.fname_Sales2019.format(city)))
            except FileExistsError:
                print("Check data_dir or check fname_Sales2019 in Config")
                break

            data2019 = pickle.load(open(os.path.join(self.data_dir, self.fname_Sales2019.format(city)),"rb"))
            data2019.sort_index(inplace=True)
            
            # Holiday
            data2019["2019-02-04"] = data2019["2019-01-28"]
            data2019["2019-02-05"] = data2019["2019-01-29"]
            data2019["2019-02-06"] = data2019["2019-01-30"]
        
            # Holiday
            data2019["2019-09-12"] = data2019["2019-09-05"]
            data2019["2019-09-13"] = data2019["2019-09-06"]
            data2019["2019-09-14"] = data2019["2019-09-07"]

            sparsity = data2019.isna().sum(axis=1) / len(data2019)
            interpolation = sparsity < 0.7
            data2019 = data2019.apply(lambda x : interpolation_na(x, "2019-12-31", 0.7), axis = 1)
            data2019 = data2019.fillna(0)
            data2019 = data2019.ewm(alpha = 0.5, min_periods = 1, axis = 1).mean()
            self._Sales2019[city] = data2019
            data_shift = data2019.shift(periods = -1, axis = "columns")

            self._SalesChange[city] = pd.DataFrame(data = ((data.iloc[:,:-2].reset_index(drop=True).values - data_shift.iloc[:,:-1].reset_index(drop=True).values) / data_shift.iloc[:,:-1].reset_index(drop=True).values), index = data.index, columns = data.iloc[:,:-2].reset_index(drop=True).columns).replace([np.inf, -np.inf], np.nan).fillna(0)

        self._lossMask = dict(sorted(self._lossMask.items()))
        self._SalesChange = dict(sorted(self._SalesChange.items()))

        self._city_dict = {v:k for k,v in enumerate(self._BusinessStructure_amt.index.values)}
        self._cat_dict = {v:k for k,v in enumerate(self._BusinessStructure_amt.columns.values)}
        ######################################################################################
        # Scaler
        ######################################################################################
        self.scaler_BusinessStructure_amt = Scaler()
        self._BusinessStructure_amt_scaled = self.scaler_BusinessStructure_amt.fit_transform(self._BusinessStructure_amt) # pd.DataFrame
        
        self.scaler_BusinessStructure_cnt = Scaler()
        self._BusinessStructure_cnt_scaled = self.scaler_BusinessStructure_cnt.fit_transform(self._BusinessStructure_cnt) # pd.DataFrame
        
        self.scaler_CustomerStructure_amt = Scaler()
        self._CustomerStructure_amt_scaled = self.scaler_CustomerStructure_amt.fit_transform(self._CustomerStructure_amt) # dict
        
        self.scaler_CustomerStructure_cnt = Scaler()
        self._CustomerStructure_cnt_scaled = self.scaler_CustomerStructure_cnt.fit_transform(self._CustomerStructure_cnt) # dict
        
        self.scaler_pdistance = Scaler()
        self._pdistance_scaled = self.scaler_pdistance.fit_transform(self._pdistance)
        
        self.scaler_cdistance = Scaler()
        self._cdistance_scaled = self.scaler_cdistance.fit_transform(self._cdistance) #
        
        self.scaler_covid_daily = Scaler()
        self._covid_daily_scaled = self._covid_daily.copy()
        self._covid_daily_scaled.iloc[:,2:] = self.scaler_covid_daily.fit_transform(self._covid_daily_scaled.iloc[:,2:]) # np.array
        
        self.scaler_covid_cum = Scaler()
        self._covid_cum_scaled = self._covid_cum.copy()
        self._covid_cum_scaled.iloc[:,2:] = self.scaler_covid_cum.fit_transform(self._covid_cum_scaled.iloc[:,2:]) # np.array
        
        self.scaler_covid_re_cum = Scaler()
        self._covid_re_cum_scaled = self._covid_re_cum.copy()
        self._covid_re_cum_scaled.iloc[:,2:] = self.scaler_covid_re_cum.fit_transform(self._covid_re_cum_scaled.iloc[:,2:]) # np.array
        
        self.scaler_covid_elapsed = Scaler()
        self._covid_elapsed_scaled = self._covid_elapsed.copy()
        self._covid_elapsed_scaled.iloc[:,2:] = self.scaler_covid_elapsed.fit_transform(self._covid_elapsed_scaled.iloc[:,2:]) # np.array
        
        self.scaler_SalesChange = SalesChangeScaler(self._lossMask)
        self._SalesChange_scaled = self.scaler_SalesChange.fit_transform(self._SalesChange) # dict

    ######################################################################################
    # Create dataset for target city & mass infection cases at startdate
    ######################################################################################
    def get_data(self, target, enddate):
        startdate_inputs = self.data_startdate
        enddate_inputs = datetime.datetime.strptime(enddate, "%Y-%m-%d")   # enddate of the inputs
 
        # enddate + 1
        startdate_target = enddate_inputs + datetime.timedelta(days = 1) # startdate of targets
        enddate_target = startdate_target + datetime.timedelta(days = self.targetPeriod - 1) # enddate of targets
        
        assert startdate_inputs < startdate_target, "startdate of inputs should be eariler than enddate of targets"
        assert target in self._BusinessStructure_amt.index , "Out of cities!"
        # startdate is for target range.
        assert startdate_target + datetime.timedelta(days = self.targetPeriod - 1) <= self.target_enddate , "Out of date!"
        
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
        ####################################################

        index_target = target
        ######################################################
        # Business structure Dataset (1)
        ######################################################
        # dataframe
        majorindustry_target = np.stack([
                                        self._BusinessStructure_amt_scaled.loc[index_target].values,
                                        self._BusinessStructure_cnt_scaled.loc[index_target].values,
                                        ], axis = -1)
        ######################################################
        # Customer structure Dataset (2)
        ######################################################
        # dictionary
        custdist_target = np.stack([
                                    self._CustomerStructure_amt_scaled[index_target].values,
                                    self._CustomerStructure_cnt_scaled[index_target].values
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
        # Epidemic View Dataset (1) - (4)
        ######################################################
        # encoder data for severity 
        severity_daily = self._covid_daily_scaled.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        severity_daily = tf.keras.preprocessing.sequence.pad_sequences(severity_daily, 
                                                                    maxlen=self.maxlen, 
                                                                    dtype='float64',
                                                                    padding = 'post')
        
        severity_cum = self._covid_cum_scaled.loc[: , startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        severity_cum = tf.keras.preprocessing.sequence.pad_sequences(severity_cum, 
                                                                    maxlen=self.maxlen, 
                                                                    dtype='float64',
                                                                    padding = 'post')
        
        severity_re_cum = self._covid_re_cum_scaled.loc[: , startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        severity_re_cum = tf.keras.preprocessing.sequence.pad_sequences(severity_re_cum, 
                                                                    maxlen=self.maxlen, 
                                                                    dtype='float64',
                                                                    padding = 'post')
        
        severity_daily = np.array(severity_daily)
        severity_cum = np.array(severity_cum)
        severity_re_cum = np.array(severity_re_cum)
        severity = np.concatenate([severity_daily[...,np.newaxis], severity_cum[...,np.newaxis], severity_re_cum[...,np.newaxis]],axis=-1) # [None, n, windowsize, 3]
        
        # decoder data for severity : [config.c, 7, 3]        
        severity_daily = self._covid_daily_scaled.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        severity_cum = self._covid_cum_scaled.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        severity_re_cum = self._covid_re_cum_scaled.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        severity_decoder = np.concatenate([severity_daily[...,np.newaxis], severity_cum[...,np.newaxis], severity_re_cum[...,np.newaxis]],axis=-1)

        # concatenate encoder + decoder data [config.c, 141, 3]
        severity = np.concatenate([severity, severity_decoder[:,:-1,:]], axis = 1) # [None, 134, 3] + [None, 6, 3]

        # Elapsed Day
        covid_elapsed_day = self._covid_elapsed_scaled.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        covid_elapsed_day_origin = self._covid_elapsed.loc[:, startdate_inputs.strftime("%Y-%m-%d"):enddate_inputs.strftime("%Y-%m-%d")].values
        covid_elapsed_day_target = self._covid_elapsed_scaled.loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        idx = np.where(covid_elapsed_day_origin[:,-1] == 0)
        covid_elapsed_day_target[idx[0], : ] = tf.tile(tf.expand_dims(covid_elapsed_day[idx[0], -1], axis = -1), [1, covid_elapsed_day_target.shape[-1]])

        covid_elapsed_day = tf.keras.preprocessing.sequence.pad_sequences(covid_elapsed_day, 
                                                                        maxlen=self.maxlen, 
                                                                        dtype='float64',
                                                                        padding = 'post')
        covid_elapsed_day = np.concatenate([covid_elapsed_day, covid_elapsed_day_target], axis = 1)

        ##############################################################################################
        # Business Structure Dataset & Customer Structure Dataset(3),(4)
        # Geography Dataset (1),(2)
        ##############################################################################################
        # generate dataset for all mass infection cases
        for i in range(index_infected.shape[0]):
            city = index_infected.loc[i, "City"]
            majorindustry_infected.append(np.stack([self._BusinessStructure_amt_scaled.loc[city].values,
                                                    self._BusinessStructure_cnt_scaled.loc[city].values,
                                                    ], axis = -1)
            )
            custdist_infected.append(np.stack([self._CustomerStructure_amt_scaled[city].values,
                                               self._CustomerStructure_cnt_scaled[city].values], axis = -1)) # dictionary
            physical_distance.append([[self._pdistance_scaled.loc[city, index_target]]])
            contextual_distance.append([[self._cdistance_scaled.loc[city, index_target]]])

        majorindustry_infected = np.array(majorindustry_infected) # [None, n, 35]
        custdist_infected = np.array(custdist_infected) # [None, n, 35, 27]

        physical_distance = np.array(physical_distance)
        contextual_distance = np.array(contextual_distance) # [None, n, 1]

        seqlen = (enddate_inputs - startdate_inputs).days + 1
        mask = tf.sequence_mask(seqlen, maxlen=self.maxlen, dtype = tf.dtypes.float64).numpy()

        #################################################
        # Target Dataset
        #################################################
        # target data
        targetSales = self._SalesChange_scaled[index_target].loc[:, startdate_target.strftime("%Y-%m-%d") : enddate_target.strftime("%Y-%m-%d")].values
        # Loss Mask
        lossMask = self._lossMask[index_target]
        # Mass Infection Mask
        covidMask = np.array((self._covid_metainfo.Startdate < enddate_inputs) & (self._covid_metainfo.Enddate + datetime.timedelta(days = 100) > enddate_inputs)).astype(float)

        #################################################
        # External Feature
        #################################################
        weekdays = [(startdate_target + datetime.timedelta(days=x)).weekday() for x in range(self.targetPeriod)] # shape = [7]
        weekdays = np.array(weekdays)

        index_target_idx = np.array([self._city_dict[index_target]])
        index_infected_idx = np.array([[self._city_dict[idx]] for idx in index_infected["City"]])
        #################################################
        # Create dataset 
        #################################################
        numOfCases = self.m
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
            )        
        metadata = np.array([index_target, startdate_target, enddate_target])
        return dataset, targetSales, lossMask, metadata

    ###################################################################################################
    # Create train dataset for all target cities & mass infection cases from 2020.02.01 ~ 2020.06.16
    ###################################################################################################
    @property
    def datasets(self):
        startdate = self.target_startdate - \
            datetime.timedelta(days = 1)
        enddate = self.target_enddate - \
            datetime.timedelta(days = self.targetPeriod - 1) # range in for loop
        target_cities = self._BusinessStructure_amt.index
        inputs = []
        labels = []
        lossMasks = []
        metadatas = []
        for ed in self.daterange(startdate, enddate):
            ed = ed.strftime("%Y-%m-%d") # enddate of inputs
            print("\rProcessing *Input* Dataset between {} ~ {}".format(self.data_startdate,ed), end="")
            for city in target_cities:
                dataset, targetSales, lossMask, metadata = self.get_data(city, ed)
                inputs.append(dataset)
                labels.append(targetSales)
                lossMasks.append(lossMask)
                metadatas.append(metadata)
        return np.array(inputs), np.array(labels), np.array(lossMasks), np.array(metadatas)

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + datetime.timedelta(n)
#%%
#######################################################################
# Training Utilities
#######################################################################
def split_data(config, datasets):
    endmonth = 11
    X, y, lossMask, metadata = datasets
    daysInMonths = [31,29,31,30,31,30,31,31,30,31,30,29]
    targetPeriod = y.shape[-1]
    daysInMonth = sum(daysInMonths[ endmonth - 1:])
    testdays = daysInMonth - targetPeriod + 1
    
    train_datasets = (X[:-(daysInMonth*25)], y[:-(daysInMonth*25)], lossMask[:-(daysInMonth*25)],  metadata[:-(daysInMonth*25)])
    covidMask = train_datasets[0][-1,:,-1]
    print(covidMask.shape)
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
    print("Test data startdate : {} ~ {}".format(test_datasets[-1][0][1], test_datasets[-1][-1][1]))
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
    print("X shape {}, y shape {}, lossMask shape {} ".format(datasets[0].shape, datasets[1].shape, datasets[2].shape))
    return datasets
