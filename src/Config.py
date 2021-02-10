#####################################################################
# Title  : Config file, hyperparameters setting
# Date   : 2020/11/19
# Update : 2021/02/08
# Author : Hyangsuk
#####################################################################
class Config():
    def __init__(
        self,

        datetime = "20210421",
        name = "COVID_EENet_RUN_TEST4",

        ## model parameters##
        w = 14,                             # prediction period 
        h = 4,                              # multi-head
        e = 20,                             # embedding dimension
        
        seq2seq_lstm_cell = 16,
        epidemicViewFCN = 1,
        geographyViewFCN = 1,
        # macroscopicAggFCN
        activation = 'tanh',

        ## train parameters##
        epochs = 20,
        batch_size = 50,                     # should be multiple of 25.
        lr = 0.001,
        weight_decay = 0.96,
        gpu_num = 0,

        ## Input file ##
        data_dir = "../data/",
        fname_BusinessStructure_amt = "BusinessStructure_AMT.pkl",                  # Real Data is Not available
        fname_BusinessStructure_cnt = "BusinessStructure_CNT.pkl",                  # Real Data is Not available
        fname_CustomerStructure = "*CustomerStructure_{}.pkl",                      # Real Data is Not available
        fname_contextual_distance = "contextual_distance_matrix.pkl",               # Real Data is Not available
        fname_physical_distance = "physical_distance.pkl",                          
        
        fname_covid_metainfo = "seoul_mass_infection_metainfo.pkl",
        fname_covid_daily = "daily_seoul_mass_infection.pkl",
        fname_covid_cum = "cumulative_seoul_mass_infection.pkl",
        fname_covid_re_cum = "recent_cumulative_seoul_mass_infection.pkl",
        fname_elapsed_day = "covid_elapsed_day.pkl",
        
        fname_Sales2020 = "*Sales2020_AMT.pkl",                                     # Real Data is Not available
        fname_Sales2019 = "{}_Sales2019_AMT.pkl",                                   # Real Data is Not available

        ## Utility ##
        start_month = 2,                     # train start month
        end_month = 11,                      # test month
        buz_dict = {'가구_offline': 0, '가전제품_offline': 1, '건강식품_offline': 2, '건축/자재_offline': 3, '광학제품_offline': 4, '기타_offline': 5,
                    '농업_offline': 6, '레져업소_offline': 7, '레져용품_offline': 8, '문화/취미_offline': 9, '보건/위생_offline': 10, '보험_offline': 11,
                    '사무/통신기기_offline': 12, '서적/문구_offline': 13, '수리서비스_offline': 14, '숙박업_offline': 15, '신변잡화_offline': 16, '여행업_offline': 17,
                    '연료판매_offline': 18, '용역서비스_offline': 19, '유통업비영리_offline': 20, '유통업영리_offline': 21, '유통업영리_online': 22, '음식료품_offline': 23,
                    '의료기관_offline': 24, '의류_offline': 25, '일반음식_offline': 26, '자동차정비/유지_offline': 27, '자동차판매_offline': 28, '주방용품_offline': 29, '직물_offline': 30,
                    '학원_offline': 31, '회원제형태업소_offline': 32, '휴게_offline': 33},
        city_dict = {'강남구': 0, '강동구': 1, '강북구': 2, '강서구': 3, '관악구': 4, '광진구': 5, '구로구': 6, '금천구': 7, '노원구': 8, '도봉구': 9, '동대문구': 10, '동작구': 11, '마포구': 12,
                     '서대문구': 13, '서초구': 14, '성동구': 15, '성북구': 16, '송파구': 17, '양천구': 18, '영등포구': 19, '용산구': 20, '은평구': 21, '종로구': 22, '중구': 23, '중랑구': 24},
        
        n = 34,                             # the number of industries, FIXED
        m = 30,                             # the number of mass infections, 
        r = 25,                             # number of regions in Seoul, FIXED
        
        ## Save File List ##
        datapath = "../data/dataset_{}_{}.pkl",
        log_filepath = '../log',
        log_name = "training_{}_{}_{}.log",
        checkpoint_filepath = '../tmp/checkpoint/{}',
        checkpoint_model = 'training_checkpoint_{}_{}_{}',
        checkpoint_outputs = "training_checkpoint_outputs_{}_{}_{}",
        
        ## ablation ##
        ablation_economicView = True,
        ablation_geographyView = True,
        ablation_macroscopicAgg = True
    ):

        ## datetime ##
        self.datetime = datetime
        self.name = name

        ## model parameters##
        self.w = w
        self.h = h 
        self.e = e

        self.seq2seq_lstm_cell = seq2seq_lstm_cell
        self.epidemicViewFCN = epidemicViewFCN
        self.geographyViewFCN = geographyViewFCN
        self.macroscopicAggFCN = w # SHOULD BE SAME WITH w
        self.activation = activation

        ## train parameters##
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gpu_num = gpu_num

        ## Input file ##
        self.data_dir = data_dir
        self.fname_BusinessStructure_amt = fname_BusinessStructure_amt
        self.fname_BusinessStructure_cnt = fname_BusinessStructure_cnt
        self.fname_CustomerStructure = fname_CustomerStructure
        self.fname_contextual_distance = fname_contextual_distance
        self.fname_physical_distance = fname_physical_distance
        self.fname_covid_metainfo = fname_covid_metainfo
        self.fname_covid_daily = fname_covid_daily
        self.fname_covid_cum = fname_covid_cum
        self.fname_covid_re_cum = fname_covid_re_cum
        self.fname_elapsed_day = fname_elapsed_day
        self.fname_Sales2020 = fname_Sales2020
        self.fname_Sales2019 = fname_Sales2019

        ## Utility ##
        self.start_month = start_month
        self.end_month = end_month
        self.buz_dict = buz_dict
        self.city_dict = city_dict
        self.m = m
        self.n = n
        self.r = r

        ## Save File List ##
        self.datapath = datapath.format(self.w, "2020-12-29")
        self.log_filepath = log_filepath
        self.log_name = log_name.format(self.datetime, self.w, self.name)
        self.checkpoint_filepath = checkpoint_filepath.format(self.name)
        self.checkpoint_model = checkpoint_model.format(self.datetime, self.w, self.name)
        self.checkpoint_outputs = checkpoint_outputs.format(self.datetime, self.w, self.name)
        
        ## ablation ##
        self.ablation_economicView = ablation_economicView
        self.ablation_geographyView = ablation_geographyView
        self.ablation_macroscopicAgg = ablation_macroscopicAgg
