class Config():
    def __init__(
        self,
        ################
        ##  datetime  ##
        ################
        ### Should be customized ###
        datetime = "20210101",
        
        ################
        ## dataloader ##
        ################
        data_dir_name = "preprocess12_new",     # preprocess, preprocess12, preprocess12_new
        data_dir = "/data/home/path-to-dir/COVIDEENet/BCCard/data/{}/",
        duration = "2020-04-09",            # data interpolation 기준 : 서울시 첫번째 유흥주점 행정명령 기준
        sparsity_threshold = 0.7,           # threshold for data interpolation or not
        n = 34,                             # the number of industries
        ### Should be customized ###
        p = 14,                             # prediction period
        ### Should be customized ###
        c = 30,                             # the number of mass infections, (2020-12-19, 30), (2020-11-30, 25)

        data_startdate = "2020-01-01",      # %Y-%m-%d, input data start date -> min: "2020-01-01" + targetSales_shift (dataloader에서 따로 계산할 예정.)
        target_startdate = "2020-02-01",    # %Y-%m-%d, target data start date -> min: "2020-01-01" + targetSales_shift + 1
        ### Should be customized ###
        target_enddate = "2020-12-29",      # %Y-%m-%d, target data end date -> max: "2020-12-31"

        buz_dict = {'가구_offline': 0, '가전제품_offline': 1, '건강식품_offline': 2, '건축/자재_offline': 3, '광학제품_offline': 4, '기타_offline': 5,
                    '농업_offline': 6, '레져업소_offline': 7, '레져용품_offline': 8, '문화/취미_offline': 9, '보건/위생_offline': 10, '보험_offline': 11,
                    '사무/통신기기_offline': 12, '서적/문구_offline': 13, '수리서비스_offline': 14, '숙박업_offline': 15, '신변잡화_offline': 16, '여행업_offline': 17,
                    '연료판매_offline': 18, '용역서비스_offline': 19, '유통업비영리_offline': 20, '유통업영리_offline': 21, '유통업영리_online': 22, '음식료품_offline': 23,
                    '의료기관_offline': 24, '의류_offline': 25, '일반음식_offline': 26, '자동차정비/유지_offline': 27, '자동차판매_offline': 28, '주방용품_offline': 29, '직물_offline': 30,
                    '학원_offline': 31, '회원제형태업소_offline': 32, '휴게_offline': 33},
        
        city_dict = {'강남구': 0, '강동구': 1, '강북구': 2, '강서구': 3, '관악구': 4, '광진구': 5, '구로구': 6, '금천구': 7, '노원구': 8, '도봉구': 9, '동대문구': 10, '동작구': 11, '마포구': 12,
                     '서대문구': 13, '서초구': 14, '성동구': 15, '성북구': 16, '송파구': 17, '양천구': 18, '영등포구': 19, '용산구': 20, '은평구': 21, '종로구': 22, '중구': 23, '중랑구': 24},

        ### Should be customized ###
        days = 60,

        ## model ##
        ### Should be customized ###
        h = 4,                              # multi-head
        ### Should be customized ###
        e = 20,                             # embedding dimension
        r = 25,                             # number of regions in Seoul
        ### Should be customized ###
        activation = 'tanh',

        ### Should be customized ###
        seq2seq_encoder_lstm_cell = 30,
        epidemic_encoder_maxlen = 50,
        outer_product_combiner = True,
        dnn_combiner = False,
        is_decoder_attn = False,
        tm_seq2seq=False,
        ablation_MAR = False,
        ablation_ECR = False,
        ablation_GER = False,
        
        
        ### Should be customized ###      
        seq2seq_decoder_lstm_cell = 16,
        ### Should be customized ###
        encoder_lstm_cell = 16,
        ### Should be customized ###
        shortTermSeverityUnits = 4,
        ### Should be customized ###
        shortTermElapsedDayUnits = 2,
        ### Should be customized ###
        longTermSeverityUnits = 3,
        ### Should be customized ###
        longTermElapsedDayUnits = 2,

        ################
        ##    train   ##
        ################
        ### Should be customized ###
        start_month = 2,
        ### Should be customized ###
        end_month = 11,
        ### Should be customized ###
        epochs = 100,
        ### Should be customized ###
        batch_size = 25,                     # should be multiple of 25.
        ### Should be customized ###
        lr = 0.01,
        ### Should be customized ###
        patience = 10,                       # early stopping
        ### Should be customized ###
        delta = 0.1,                         # early stopping

        ####################
        ## Save File List ##
        ####################
        name = "",
        log_filepath = '/data/home/path-to-dir/COVIDEENet/BCCard/src/log',
        log_name = "training_{}_{}_{}.log",
        datapath = "/data/home/path-to-dir/COVIDEENet/BCCard/src/data/{}_dataset_{}_{}_{}.pkl",
	initializer_path = "/data/home/path-to-dir/COVIDEENet/BCCard/src/initializer/initializer_{}_{}_",
        checkpoint_filepath = '/data/home/path-to-dir/COVIDEENet/BCCard/src/checkpoint/{}',
        checkpoint_model = '{}_{}_{}',
        checkpoint_state = 'state_{}_{}_{}',
        checkpoint_outputs = "outputs_{}_{}_{}",
        checkpoint_figures = "figures_{}_{}_{}",
        checkpoint_figurespath = '/data/home/path-to-dir/COVIDEENet/BCCard/src_torch/img/{}',
        
        ##############
        ## ablation ##
        ##############
    ):
        ## datetime ##
        self.datetime = datetime

        ## dataloader ##
        self.data_dir_name = data_dir_name
        self.data_dir = data_dir.format(self.data_dir_name)
        self.duration = duration
        self.sparsity_threshold = sparsity_threshold
        self.p = p
        self.n = n
        self.c = c
        self.data_startdate = data_startdate
        self.target_startdate = target_startdate 
        self.target_enddate = target_enddate  

        self.buz_dict = buz_dict
        self.city_dict = city_dict
        self.days = days

        self.start_month = start_month
        self.end_month = end_month
        
        ## model ##
        self.h = h 
        self.e = e 
        self.r = r
        self.activation = activation
        self.seq2seq_encoder_lstm_cell = seq2seq_encoder_lstm_cell
        self.seq2seq_decoder_lstm_cell = 16
        self.epidemic_encoder_maxlen = epidemic_encoder_maxlen
        self.outer_product_combiner = outer_product_combiner
        self.dnn_combiner = dnn_combiner
        self.tm_seq2seq = tm_seq2seq
        self.is_decoder_attn = is_decoder_attn
        self.ablation_MAR = ablation_MAR
        self.ablation_ECR = ablation_ECR
        self.ablation_GER = ablation_GER
        
        ## train ##
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.delta = delta
        self.patience = patience

        ## Save File List ##
        self.name = name
        self.log_filepath = log_filepath
        self.log_name = log_name.format( self.name,  self.p, self.datetime)
        self.datapath = datapath.format(self.data_dir_name, self.p, self.days, self.target_enddate)
        self.initializer_path = initializer_path.format(self.datetime,  self.p)
        self.checkpoint_filepath = checkpoint_filepath.format(self.name)
        self.checkpoint_model = checkpoint_model.format( self.name,  self.p, self.datetime)
        self.checkpoint_state = checkpoint_state.format( self.name,  self.p, self.datetime)
        self.checkpoint_outputs = checkpoint_outputs.format( self.name,  self.p, self.datetime)
        
        self.checkpoint_figures = checkpoint_figures.format( self.name, self.p, self.datetime)
        self.checkpoint_figurespath = checkpoint_figurespath.format(self.name)
        
        ## ablation ##
