#####################################################################
# Title  : Preprocess code for economic activity dataset from BCCard
# Date   : 2020/11/19
# Update : 2021/02/08
# Author : Hyangsuk
#####################################################################
#%%
import pandas as pd
import numpy as np
import os, pickle

#%%
tp_grp_nm_dict = {"유흥주점(음식)":"휴게","단란주점(음식)":"휴게"}
def SaleDistForCity(data, sales = 'CNT', total=False):
    if total :
        # get the sales distribution for seoul
        table = data[["TP_GRP_NM",sales]].groupby("TP_GRP_NM",as_index=True).agg('sum')
        table.fillna(0,inplace=True)
        table[sales] = table[sales] / table[sales].sum()
        return table.T
    else :    
        # get the sales distribution for each city
        table = pd.pivot_table(data,index="CTY_RGN_NM",columns="TP_GRP_NM",values=sales,aggfunc='sum')
        table.fillna(0,inplace=True)
        table = (table.T / table.sum(axis=1)).T
        return table

def _ReLU(data):
    data[data < 0 ] = 0
    return data

def _Positive_Softmax(data):
    data = (np.exp(data[data > 0]).fillna(0).T / np.exp(data[data > 0]).sum(axis=1)).T
    return data

def SaleDiff(data_city, data_total):
    # subtract the sales distribution for each city with the sales di
    data = (data_city.values - data_total.values) / data_total.values
    data = pd.DataFrame(data = data, index = data_city.index, columns = data_city.columns)
    data = _ReLU(data)
    data = _Positive_Softmax(data)
    return data

def StoreDiff_load(data_dir):
    # MAJOR INDUSTRY : NUM OF CHOPS, NORMALIZED NUM OF SHOPS
    data = pickle.load(open(os.path.join(data_dir,"BusinessStructure_AMT.pkl"),"rb"))
    tp_grp_nm = data.columns
    city = data.index

    tmp = pickle.load(open(os.path.join(data_dir, "shops_cnt_df.pkl"),"rb"))
    cnt_mat = tmp.reset_index(level=[0,1],)
    cnt_mat["상권업종중분류명"] = cnt_mat["상권업종중분류명"].replace(tp_grp_nm_dict)
    cnt_mat = cnt_mat.pivot(index=["시군구명"], columns=["상권업종중분류명"], values=0)
    cnt_mat = (cnt_mat.T / cnt_mat.sum(axis=1)).T
    
    norm_cnt_mat = cnt_mat.div(cnt_mat.sum(axis=1), axis='index')
    norm_cnt_mat = (norm_cnt_mat.T / norm_cnt_mat.sum(axis=1)).T

    cnt_mat.columns = cnt_mat.columns + "_offline"
    norm_cnt_mat.columns = norm_cnt_mat.columns + "_offline"
    
    cnt_mat = cnt_mat.reindex(tp_grp_nm, fill_value = 0, axis = 1).reindex(city, fill_value = 0, axis = 0)
    cnt_mat_total = cnt_mat.mean(axis=0)
    cnt_mat_diff = SaleDiff(cnt_mat, cnt_mat_total)
    cnt_mat_diff.fillna(0,inplace=True)

    fname = "majorIndustry_numOfShopCNT.pkl"
    fname = os.path.join(data_dir, fname)
    save_pickle(cnt_mat_diff, fname)
    
    norm_cnt_mat = norm_cnt_mat.reindex(tp_grp_nm, fill_value = 0, axis = 1).reindex(city, fill_value = 0, axis = 0)
    norm_cnt_mat_total = norm_cnt_mat.mean(axis=0)
    norm_cnt_mat_diff = SaleDiff(norm_cnt_mat, norm_cnt_mat_total)
    norm_cnt_mat_diff.fillna(0,inplace=True)

    fname = "majorIndustry_numOfShopNORM.pkl"
    fname = os.path.join(data_dir, fname)
    save_pickle(norm_cnt_mat_diff, fname)

def CustSaleDistForCity(data, city, sales = 'CNT'):
    # get the sales distribution for each customer
    assert city in data.CTY_RGN_NM.unique()
    tp_grp_nm = data.TP_GRP_NM.unique()
    data["cust_type"] = data["AGE_VAL"].apply(str)+"_"+data["SEX_CTGO_CD"].apply(str)+"_"+data["FLC"].apply(str)
    cust_type = data.cust_type.unique()
    data = data[data.CTY_RGN_NM == city]
    data = pd.pivot_table(data, values = sales, index = "TP_GRP_NM", columns = "cust_type", aggfunc = "sum", fill_value = 0)\
        .reindex(cust_type, fill_value = 0, axis = 1).reindex(tp_grp_nm, fill_value = 0, axis = 0)
    data.fillna(0,inplace=True)
    data = (data.T / data.sum(axis=1)).T
    return data

def TargetSales(data, city, sales = "AMT") :
    assert city in data.CTY_RGN_NM.unique()
    tp_grp_nm = data.TP_GRP_NM.unique()
    sale_date = data.SALE_DATE.unique()
    sale_date.sort()
    data = data[data.CTY_RGN_NM == city]
    data = pd.pivot_table(data, values = sales, index = "TP_GRP_NM", columns = "SALE_DATE", aggfunc = "sum")\
        .reindex(sale_date, axis = 1).reindex(tp_grp_nm, axis = 0)
#     data.fillna(0, inplace=True)
    data.columns = pd.to_datetime(data.columns, format="%Y%m%d")
    return data

def TargetSalesDiff365(city, data_dir):
    data2019 = pickle.load(open(os.path.join(data_dir, "{}_SalesChange_2019_AMT.pkl".format(city)),"rb"))
    data2019.sort_index(inplace=True)
    data2020 = pickle.load(open(os.path.join(data_dir, "{}_SalesChange_2020_AMT.pkl".format(city)),"rb"))
    data2020.sort_index(inplace=True)
    data_shift = data2019.shift(periods = -1, axis="columns") # shift p days
    return pd.DataFrame(data = (data2020.values- data_shift.values) / data_shift.values, columns = data2020.columns, index= data2020.index)

def load_pickle(months=[13,18], reduced = True, online = True):
    fnames = [201901,201902,201903,201904,201905,201906,201907,201908,201909,\
        201910,201911,201912,202001,202002,202003,202004,202005,202006,202007,202008,202009,202010,202011,202012]
    st = int(months[0])
    ed = int(months[-1])
    print("Preprocessing {} ~ {}".format(fnames[st-1],fnames[ed-1]))
    fname = "../../data/seoul_pickle/seoul_{}.pkl".format(str(fnames[st-1]))
    print(fname)
    with open(fname,"rb") as f:
        df = pickle.load(f)
    for i in fnames[st:ed]:
        fname = "../../data/seoul_pickle/seoul_{}.pkl".format(str(i))
        try:
            print(fname)
            with open(fname,"rb") as f:
                data = pickle.load(f)
                df = pd.concat([df,data])
                del data
        except FileExistsError:
            continue
    if reduced:
        df["TP_GRP_NM"] = df["TP_GRP_NM"].replace(tp_grp_nm_dict)
    if online:
        df["TP_GRP_NM"] = df["TP_GRP_NM"] + df["ON2_OFFLINE1"].replace({1:"_offline",2:"_online"})
    return df

def save_pickle(data,fname):
    with open(fname,"wb") as f:
        pickle.dump(data,f)
    print("Save at {}".format(fname))

#%%
def load_data_past(d = [1,12]):
    data_past = load_pickle(months = d)
    return data_past

def load_data_present(d = [13,24]):
    data_present = load_pickle(months = d)
    return data_present

def preprocess_dataset(data_past):
    data_dir = os.path.join(os.getcwd(),"../data/")
    print("Save data at {}".format(data_dir))
    ###################################################################################
    # Num of Shops          : static
    # Business Structure    : 2019.01 ~ 2019.12
    # Customer Structure    : 2019.01 ~ 2019.12
    # Sales Change          : 2020.01 ~ 2020.12
    ###################################################################################
    if data_past is not None:
        # Business Structure : AMT, CTN
        data_city = SaleDistForCity(data_past, sales="CNT", total=False)
        data_total = SaleDistForCity(data_past, sales="CNT", total=True)
        majorIndustry = SaleDiff(data_city, data_total)
        fname = "BusinessStructure_CNT.pkl"
        fname = os.path.join(data_dir, fname)
        save_pickle(majorIndustry, fname)

        data_city = SaleDistForCity(data_past, sales="AMT", total=False)
        data_total = SaleDistForCity(data_past, sales="AMT", total=True)
        majorIndustry = SaleDiff(data_city, data_total)
        fname = "BusinessStructure_AMT.pkl"
        fname = os.path.join(data_dir, fname)
        save_pickle(majorIndustry, fname)

        # Customer Structure : AMT, CTN
        cities = data_past.CTY_RGN_NM.unique()
        for city in cities:
            data_city = CustSaleDistForCity(data_past, city, sales="AMT")
            fname = "{}_CustomerStructure_AMT.pkl".format(city)
            fname = os.path.join(data_dir, fname)
            save_pickle(data_city,fname)
            
            data_city = CustSaleDistForCity(data_past, city, sales="CNT")
            fname = "{}_CustomerStructure_CNT.pkl".format(city)
            fname = os.path.join(data_dir, fname)
            save_pickle(data_city,fname)

def preprocess_dataset_target(data_present, year):
    data_dir = os.path.join(os.getcwd(),"../data")
    # data_present = load_pickle(months = [13,24])
    data = pickle.load(open(os.path.join(data_dir,"BusinessStructure_AMT.pkl"),"rb"))
    tp_grp_nm = data.columns
    cities = data_present.CTY_RGN_NM.unique()
    for city in cities:
        print(city, end= "\t")
        data_city = TargetSales(data_present, city, sales="AMT")
        fname = "{}_Sales{}_AMT.pkl".format(city, year)
        fname = os.path.join(data_dir, fname)
        save_pickle(data_city,fname)
        
        data_city = TargetSales(data_present, city, sales="CNT")
        fname = "{}_Sales{}_CNT.pkl".format(city, year)
        fname = os.path.join(data_dir, fname)
        save_pickle(data_city,fname)
#%%
# preprocess_dataset()
if __name__ == '__main__':
    print("preprocessing")
    data_past = load_data_past(d = [1,12])
    preprocess_dataset(data_past)
    preprocess_dataset_target(data_past, 2019)
del data_past
#%%
if __name__ == '__main__':
    print("preprocessing")
    data_present = load_data_present(d = [13,24]) # 202001 ~ 202012
    preprocess_dataset_target(data_present, 2020)
#%%
if __name__ == '__main__':
    data_dir = "../data"
    StoreDiff_load(data_dir)
