import torch
import torch.nn as nn
import numpy as np
import pandas as pd

covid_mask = torch.load("data/covid_mask.pt")
lossmask = torch.load("data/lossmask_district_buz.pt")
industry_list = torch.load("data/industry_list.pt")
city_dict = torch.load("data/city_dict.pt")
mass_inf_info = torch.load("data/mass_inf_info.pt")

def get_TCN_baseline(config):
    try:
        from models.tcn import TCN_FCN
        return TCN_FCN(config)
    except:
        return
    
def get_TADA_baseline(config):
    try: 
        from models.tada import TADA
        return TADA(config)
    except:
        return
    
def get_DEFSI_baseline(config):
    try: 
        from models.defsi import DEFSI
        return DEFSI(config)
    except:
        return
    
def get_Seq2SeqATTN_baseline(config):
    try: 
        from models.seq2seqattn import Seq2SeqATTN
        return Seq2SeqATTN(config)
    except:
        return
    
def get_COVIDEENet(config):
    try: 
        from models.covideenet import COVIDEENet
        return COVIDEENet(config)
    except:
        return       
    
def check_perf_five_models(model_name, testloader, directory, model_state_dict_fname, config):    
    
    rmse_list, mae_list = [], []
    for i in range(5):
        model, val_x, rmse_tot, mae_tot = load_perf_model(model_name, testloader,
                                                          directory, 
                                                          model_state_dict_fname.format(i), 
                                                          config)
        print(i+1,": ", rmse_tot, mae_tot)
        rmse_list.append(rmse_tot)
        mae_list.append(mae_tot)
    
    rmse_list, mae_list = torch.tensor(rmse_list), torch.tensor(mae_list)
    best_model_id = rmse_list.argmin()
    model_state_dict_fname = model_state_dict_fname.format(best_model_id)
    model, val_x, _, _ = load_perf_model(model_name, testloader, directory, model_state_dict_fname, config)
    
    rmse_mean, rmse_std = rmse_list.mean(), rmse_list.std()
    mae_mean, mae_std = mae_list.mean(), mae_list.std()
    print()
    print("RMSE {} +- {}".format(rmse_mean, rmse_std))
    print("MAE {} +- {} \n".format(mae_mean, mae_std))
    
    return model, val_x
    
def load_perf_model(model_name, testloader, directory, model_state_dict_fname, config):
    if model_name.lower() == "covideenet":
        model = get_COVIDEENet(config).to(config.device)
    if model_name.lower() == "tcn":
        model = get_TCN_baseline(config).to(config.device)
    if model_name.lower() == "tada":
        model = get_TADA_baseline(config).to(config.device)
    if model_name.lower() == "seq2seqattn":
        model = get_Seq2SeqATTN_baseline(config).to(config.device)
    if model_name.lower() == "defsi":
        model = get_DEFSI_baseline(config).to(config.device)
        
    if model_name.lower() == "covideenet":
        best_model_state_dict = torch.load(directory+model_state_dict_fname)
    else:
        best_model_state_dict = torch.load(directory+model_state_dict_fname)
    model.load_state_dict(best_model_state_dict)
    
    rmse_tot, mae_tot, _, _, _, val_x = perf(model, testloader, model_name, config)
    
    return model, val_x, rmse_tot, mae_tot

def perf(model, testloader, model_name, config, inspect=False):
    rmse_list, mae_list = [], []
    with torch.no_grad():
        for val_x in testloader: # testloader, valloader
            if model_name.lower() == "covideenet":
                mseloss, y_hat, modeling_output = model(val_x, inspect=inspect, )
                val_y_truth = val_x[1].to(config.device)
            else:
                mseloss, y_hat, modeling_output = model(val_x, inspect=inspect, )
                val_y_truth = val_x[-1].to(config.device)
            maeloss = nn.L1Loss(reduction='none')(y_hat, 
                                                  val_y_truth.to(torch.float32))
            total_rmse = torch.sqrt((mseloss.cpu() * lossmask.cpu().unsqueeze(-1)).mean())
            total_mae = (maeloss.cpu() * lossmask.unsqueeze(-1)).mean()
            rmse_list.append(total_rmse)
            mae_list.append(total_mae)

    if len(rmse_list) > 1:
        rmse_list, mae_list = torch.stack(rmse_list, dim=-1), torch.stack(mae_list, dim=-1) # (bs, ind, pred)
        rmse_list = rmse_list.mean(-1)
        mae_list = mae_list.mean(-1)

    rmse_tot = torch.tensor(rmse_list).mean()
    mae_tot = torch.tensor(mae_list).mean()
    
    return rmse_tot, mae_tot, modeling_output, y_hat, val_y_truth, val_x

def save_model_prediction(model_name, 
                          testloader,
                          directory,
                          model_state_dict_fname,
                          config, 
                          result_save_directory):
    
    for i in range(5):
        fname = model_state_dict_fname.format(i)
        model, val_x, _, _ = load_perf_model(model_name=model_name,
                                             testloader=testloader,
                                             directory=directory,
                                             model_state_dict_fname=fname,
                                             config=config)
        _, _, _, y_hat, val_y_truth, val_x = perf(model, testloader,
                                                  model_name=model_name, 
                                                  config=config,
                                                  inspect=True)
        from os import getcwd
        root_directory = getcwd()
        torch.save((val_y_truth.cpu().numpy(), y_hat.cpu().numpy()),
                   "{}/{}/{}_{}.pt".format(root_directory, result_save_directory, model_name, i))


def save_results_district_buz_pair(config, 
                                   testloader, 
                                   model_name,
                                   model_directory,
                                   result_save_directory,
                                   model_state_dict_fname, 
                                   lossmask,
                                   industry_list, 
                                   city_list):
    if model_name.lower() == "covideenet":
        model = get_COVIDEENet(config).to(config.device)
    if model_name.lower() == "tcn":
        model = get_TCN_baseline(config).to(config.device)
    if model_name.lower() == "tada":
        model = get_TADA_baseline(config).to(config.device)
    if model_name.lower() == "seq2seqattn":
        model = get_Seq2SeqATTN_baseline(config).to(config.device)
    if model_name.lower() == "defsi":
        model = get_DEFSI_baseline(config).to(config.device)

    model_state_dict = torch.load(model_directory+model_state_dict_fname)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    with torch.no_grad():
        for val_x in testloader:
            if model_name.lower() == "covideenet":
                mseloss, y_hat, modeling_output = model(val_x, inspect=False)
                val_y_truth = val_x[1].to(config.device)
                maeloss = nn.L1Loss(reduction='none')(y_hat, 
                                                      val_y_truth.to(torch.float32))
            else:
                mseloss, y_hat, modeling_output = model(val_x, inspect=False, )
                val_y_truth = val_x[-1].to(config.device)
                maeloss = nn.L1Loss(reduction='none')(y_hat, 
                                                      val_y_truth.to(torch.float32))
    # mseloss: (region, ind, pred)
    lossmask = lossmask.to(config.device)
    pairRMSE = torch.sqrt((((mseloss) * lossmask.unsqueeze(-1))).mean(dim=-1))
    pairMAE = torch.sqrt((((maeloss) * lossmask.unsqueeze(-1))).mean(dim=-1))

    model_rmseDF = pd.DataFrame(pairRMSE.cpu().numpy(),
                                     index = city_list, 
                                     columns= industry_list)
    from os import getcwd
    root_directory = getcwd()
    model_rmseDF.to_excel("{}/{}/{}.xlsx".format(root_directory,
                                                      result_save_directory,
                                                      model_state_dict_fname[:-3]))
    return model_rmseDF



def get_rmse_mean_std(config, 
                      model_name,
                      testloader,
                      model_directory,
                      result_save_directory, 
                      model_state_dict_fname, 
                      industry_list,
                      city_list):
    model_rmseDF_list = []
    for i in range(5):
        model_rmseDF = save_results_district_buz_pair(config, testloader,
                                                      model_name=model_name,
                                      model_directory=model_directory,
                                      result_save_directory=result_save_directory,
                                      model_state_dict_fname=model_state_dict_fname.format(i), 
                                      lossmask=lossmask,
                                      industry_list=industry_list,
                                      city_list=city_list)
        model_rmseDF_list.append(model_rmseDF)
        
    model_rmse_mean_std = np.stack(model_rmseDF_list, axis=-1)
    model_rmse_mean = model_rmse_mean_std.mean(-1)
    model_rmse_std = model_rmse_mean_std.std(-1)
    
    model_rmse_mean = pd.DataFrame(model_rmse_mean,
                                        index = city_list, 
                                        columns= industry_list)
    model_rmse_std = pd.DataFrame(model_rmse_std,
                                       index = city_list, 
                                       columns= industry_list)
    from os import getcwd
    root_dir = getcwd()
    model_rmse_mean.to_excel("{}/{}/{}.xlsx".format(root_dir,
                                                      result_save_directory,
                                                      model_name+"_mean"))
    model_rmse_std.to_excel("{}/{}/{}.xlsx".format(root_dir,
                                                      result_save_directory,
                                                      model_name+"_std"))

        
def make_input_for_epidemic_encoder(model, bs, valid_len, weekdays, covid_elapsed_day,
                           epidemiological_severity, covid_mask, covid_start,):
    epidemiological_severity = epidemiological_severity.to(torch.float32)
    is_variablelen = valid_len[0] < model.max_len + 1

    ## Processing valid indice #################################### 
    if is_variablelen: # for variable length
        uniq_val, inv_ind = valid_len.unique(return_inverse=True)
        uniq_val = uniq_val.to(torch.long)
        uniq_val_dict = {_: torch.arange(uniq_val[_]) for _ in range(uniq_val.size(0))}
        gather_ind = torch.nn.utils.rnn.pad_sequence([uniq_val_dict[l.item()] for l in inv_ind],
                                                     batch_first = True,
                                                     padding_value = -1.).to(torch.long) # (bs, model.var_len:e.g.34)

    else:
        gather_ind = torch.cat([valid_len.unsqueeze(-1) - _ - 1 for _ in range(model.max_len-1,-1,-1)], 
                               dim=-1).to(torch.long) # (bs, model.max_len=50)
    gather_ind_dec = torch.cat([valid_len.unsqueeze(-1) + _ for _ in range(0-1,model.pred_len,1)], 
                               dim=-1).to(torch.long) # (bs, model.pred_len=1+14 or 1+28)
    ####################### ####################### ####################### #######

    ## Processing weekdays_enc ####################################
    week_rep = (valid_len[-1]//model.config.p+1).to(torch.long)
    if is_variablelen: # for variable length 
        weekdays_enc = torch.cat([weekdays for i in range(week_rep)], 
                                 axis=1) # (bs, var_len(e.g. 34), emb_dim=6)
        weekdays_enc_dict = {_:weekdays_enc[model.region*_,-uniq_val[_]:,:] for _ in range(uniq_val.size(0))} # (bs, var_len(e.g. 34), emb_dim=6)
        weekdays_enc = torch.nn.utils.rnn.pad_sequence([weekdays_enc_dict[l.item()] for l in inv_ind],
                                                       batch_first = True,
                                                       padding_value = 0.).to(torch.long) # (bs, var_len:e.g.34,emb_dim)
        weekdays_enc = weekdays_enc.unsqueeze(2).repeat(1, 1, model.numOfMassInfection,1) 
        # weekdays_enc: (bs,var_len,#massinf,emb_dim=6)

    else: 
        weekdays_enc = torch.cat([weekdays for i in range(week_rep)], 
                             axis=1)[:,-model.max_len:,:] # (bs, model.max_len=50, emb_dim=6)
        weekdays_enc = weekdays_enc.unsqueeze(2).repeat(1, 1, model.numOfMassInfection,1) 
        # weekdays_enc: (bs,model.max_len,#massinf,emb_dim=6)

    weekdays_dec = torch.cat([weekdays for i in range(model.pred_len//model.config.p+1)], 
                         axis=1)[:,:model.pred_len,:] # (bs, model.pred_len=14or28, emb_dim=6)
    weekdays_dec = torch.cat([weekdays[:,[-1],:],weekdays_dec], dim=1) # (bs, 1+model.pred_len, emb_dim=6)
    weekdays_dec = weekdays_dec.unsqueeze(2).repeat(1, 1, model.numOfMassInfection, 1) 
    # (bs, model.pred_len,#massinf,emb_dim)
    ####################### ####################### ####################### #######

    ## Processing covid_elapsed_enc processing w/ gather_ind ####################################
    covid_elapsed_enc = covid_elapsed_day[torch.arange(bs).unsqueeze(-1), gather_ind] 
    # (bs, model.max_len or var_len(e.g.34), #massinf, dim=6)
    covid_elapsed_dec = covid_elapsed_day[torch.arange(bs).unsqueeze(-1), gather_ind_dec]
    # (bs, model.pred_len, #massinf, dim=6)
    #############################################################################################

    ## Processing epidemiological_severity_enc w/ gather_ind ####################################
    epidemiological_severity_enc = epidemiological_severity[torch.arange(bs).unsqueeze(-1), gather_ind]
    # (bs, model.max_len or var_len(e.g.34), #massinf, n_feats)
    epidemiological_severity_dec = epidemiological_severity[torch.arange(bs).unsqueeze(-1), gather_ind_dec]
    # (bs, 1+model.pred_len, #massinf, n_feats)
    #############################################################################################

    inp_enc_cat = torch.cat([weekdays_enc, covid_elapsed_enc, epidemiological_severity_enc], 
                            dim=-1) # (bs, model.max_len or var_len(e.g.34), #massinf, n_feats * 3)
    #### init input_len for pack_padded_sequence #######
    if is_variablelen:
        input_len = valid_len.unsqueeze(-1).repeat(1, model.numOfMassInfection) # (bs, #massinf)
    else:
        input_len = inp_enc_cat.size(1)*torch.ones(bs, model.numOfMassInfection) # (bs, #massinf)
        input_len = input_len.to(model.config.device)

    # active mass infections
    active_inf = torch.unique(covid_mask.nonzero(as_tuple=True)[1])

    ###### comment ########################
    # only when there are active mass infs
    if active_inf.nelement() != 0:
        active_inf_start = torch.tensor([covid_start[m_inf.item()] for m_inf in active_inf]) + 31
        active_inf_start = active_inf_start.to(model.config.device)
        active_inf_dt = inp_enc_cat[:,:,active_inf,:] # (bs, model.max_len or var_len(e.g.34), #active massinf, n_feats * 3)
        active_inf_dt = active_inf_dt.transpose(1,2).contiguous().view(-1, inp_enc_cat.size(1), inp_enc_cat.size(3))
        # (bs * #active massinf,  model.max_len or var_len(e.g.34), n_feats * 3)

        slice_len = (valid_len - active_inf_start.unsqueeze(-1) + 1).T # days_past_outbreak: (bs, #active massinf)
        cut_ind = ((slice_len > 0) * (slice_len < model.max_len+1)) 
        # 0<days_past_outbreak<51; otherwise, will be masked(<0) or already got the recent_data =model.max_len
        slice_len = slice_len.reshape(-1).to(torch.long) # (bs * #active massinf)
        cut_ind = cut_ind.reshape(-1) # (bs * #active massinf)

        if cut_ind.sum() > 0 : # prevent cut_ind full of False
            if is_variablelen:
                num_gatherind_pads = (gather_ind < 0).sum(1).unsqueeze(-1) #(bs, var_len -> 1)
                num_gatherind_pads = num_gatherind_pads.unsqueeze(1).repeat(1, model.numOfMassInfection, 1)
                num_gatherind_pads = num_gatherind_pads[:,active_inf,:] #(bs, #mass->#active_mass, 1)
                num_gatherind_pads = num_gatherind_pads.view(-1) #(bs * #active_mass)

                padded_dt = torch.nn.utils.rnn.pad_sequence([row[:-num_gatherind_pads[i]][cut:] for i, (row, cut) in enumerate(zip(active_inf_dt[cut_ind], - slice_len[cut_ind]))],
                                                            batch_first=True, padding_value=0.) # (none, active_duration, cat_feats)

            else:
                padded_dt = torch.nn.utils.rnn.pad_sequence([row[cut:] for row, cut in zip(active_inf_dt[cut_ind], - slice_len[cut_ind])],
                                                        batch_first=True, padding_value=0.) # (none, active_duration, cat_feats)
            pad_of_padded_dt = torch.zeros(padded_dt.size(0),inp_enc_cat.size(1)-padded_dt.size(1),padded_dt.size(2))
            pad_of_padded_dt = pad_of_padded_dt.to(model.config.device)
            padded_dt = torch.cat((padded_dt, pad_of_padded_dt),
                                  dim=1) # (none, model.max_len or var_len, cat_feats)
            active_inf_dt[cut_ind] = padded_dt  # (bs * #active massinf, model.max_len or var_len, n_feats * 3)
            active_inf_dt = active_inf_dt.view(bs, -1, inp_enc_cat.size(1), inp_enc_cat.size(3)).contiguous().transpose(1,2) 
            # (bs,  model.max_len or var_len, #active massinf, cat_feats)

            inp_enc_cat[:,:,active_inf,:] = active_inf_dt # (bs, model.max_len or var_len, #massinf, cat_feats)

            ##### correct input_len ########################################################
            slice_len = slice_len.reshape(-1).to(torch.float32)
            act_input_len = input_len[:,active_inf].view(-1) # (bs * #active massinf)

            act_input_len[cut_ind] = slice_len[cut_ind]
            input_len[:, active_inf] = act_input_len.view(bs, -1)

    ###### comment ########################
    inp_enc_cat = inp_enc_cat.transpose(1,2).contiguous().view(-1, inp_enc_cat.size(1), inp_enc_cat.size(-1))
    # (bs * #massinf,  model.max_len or valid_len /  var_len, cat_feats)
    input_len = input_len.reshape(-1) # (bs * #massinf)

    weekdays_dec = weekdays_dec.transpose(1,2).contiguous().view(bs*model.numOfMassInfection,
                                                                1+model.pred_len, 
                                                                 weekdays_dec.size(-1))
    covid_elapsed_dec = covid_elapsed_dec.transpose(1,2).contiguous().view(bs*model.numOfMassInfection,
                                                                          1+model.pred_len, 
                                                                           covid_elapsed_dec.size(-1))
    epidemiological_severity_dec = epidemiological_severity_dec.transpose(1,2).contiguous().view(bs*model.numOfMassInfection,
                                                                                                1+model.pred_len, 
                                                                                                 epidemiological_severity_dec.size(-1))


    return (inp_enc_cat,input_len), \
            weekdays_dec,\
            covid_elapsed_dec,\
            epidemiological_severity_dec