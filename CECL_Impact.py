#!/usr/bin/env python
# coding: utf-8

# In[85]:

import requests
import pandas as pd
import numpy as np
import math
import scipy
from scipy.stats import norm
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from copy import deepcopy
import base64
import io


# In[74]:


# Normallize the rating migration matrix for each segment so that each row adds up to 1
def rel_mat(seg, TM):
    rel_mat_t_list = []
    for s in range(len(seg)):
        rel_mat_t = TM[seg[s]].drop(["rating_alph"], axis=1)
        for j in range(rel_mat_t.shape[1]):
            rel_mat_t.iloc[:,j] = rel_mat_t.iloc[:,j]/rel_mat_t.sum(axis=1)
        rel_mat_t_list.append(rel_mat_t)
    return rel_mat_t_list


# In[75]:


# Keep the MEVs used in regression and generate lags up to 2 quarters
def MacroTrans(macro):
    period_len = macro.shape[0]
    macro["period"] = pd.date_range('1976-03-01', periods=period_len, freq='Q')
    
    macro = macro[macro.period > '1989-12-31']

    macro = macro[["period", "Real GDP growth", "Unemployment rate", "10-year Treasury yield", "BBB corporate yield", 
                   "Dow Jones Total Stock Market Index (Level)", "Market Volatility Index (Level)", "WTI Price",
                   "Euro area real GDP growth"]]
    macro.columns = ["period", "usgdp", "ur", "tenyr_yield", "bbb_yield", "djix", "vix", "wti", "eugdp"]
    macro["spread"] = macro["bbb_yield"] - macro["tenyr_yield"]
    macro["djix_lag4"] = macro["djix"].shift(periods=4)
    macro["djgr"] = macro["djix"]/macro["djix_lag4"] - 1
    macro["vix_lag4"] = macro["vix"].shift(periods=4)
    macro["vixgr"] = macro["vix"]/macro["vix_lag4"] - 1
    macro["wti_lag4"] = macro["wti"].shift(periods=4)
    macro["wtigr"] = macro["wti"]/macro["wti_lag4"] - 1
    

    macro = macro[["period", "usgdp", "ur", "spread", "djgr", "vixgr", "wtigr", "eugdp"]]
    colname = macro.columns.values
    for i in range(1,len(colname)):
        macro[colname[i]+"_lag1"] = macro[colname[i]].shift(periods=1)
        macro[colname[i]+"_lag2"] = macro[colname[i]].shift(periods=2)
        macro.rename(columns={colname[i]: colname[i]+"_lag0"}, inplace=True)

    macro.dropna(inplace=True)
    return macro


# In[76]:


# Generate the shift factors for all segments in the forecast horizon
def shift_list(DR, macro_hist, macro_scenario):
    # Generate segment level default rate data
    segment2 = ["Fin_US","Oil_US","Cyc_US1","Cyc_US2","NCy_US1","NCy_US2"]
    PDlist = {}

    PDlist["Fin_US"]  = DR["Fin_US"][["period","PD1"]].copy()
    PDlist["Oil_US"]  = DR["Oil_US"][["period","PD1"]].copy()
    PDlist["Cyc_US1"] = DR["Cyc_US_2clst"][["period","PD2"]].copy()
    PDlist["Cyc_US2"] = DR["Cyc_US_2clst"][["period","PD3"]].copy()
    PDlist["NCy_US1"] = DR["Cyc_US_2clst"][["period","PD2"]].copy()
    PDlist["NCy_US2"] = DR["NCy_US"][["period","PD4"]].copy()

    # Define MEVs used in each of the segments
    MEVlist = {}
    MEVlist["Fin_US"] = ["usgdp_lag0", "spread_lag1"]
    MEVlist["Oil_US"] = ["vixgr_lag0", "wtigr_lag1"]
    MEVlist["Cyc_US1"] = ["usgdp_lag0", "djgr_lag0"]
    MEVlist["Cyc_US2"] = ["usgdp_lag0", "spread_lag1"]
    MEVlist["NCy_US1"] = ["usgdp_lag0", "djgr_lag0"]
    MEVlist["NCy_US2"] = ["spread_lag2", "vixgr_lag0"]

    # Run the regression model of historical default rate on MEVs and generate PD forecast to derive the shift factor
    Shiftlist = {}
    for s in range(len(segment2)):
        print(segment2[s])
        # Prepare data for regression by merging the MEVs with historical default rate
        PDlist[segment2[s]]["period"] = pd.date_range(PDlist[segment2[s]]["period"][0], periods=PDlist[segment2[s]].shape[0], freq='Q')
        PDlist[segment2[s]].rename(columns={PDlist[segment2[s]].columns[1]: "PD"}, inplace=True)
        RegData_final = PDlist[segment2[s]].merge(macro_hist, left_on="period", right_on="period")
        RegData_final.dropna(inplace=True)

        # Run fractional logit regression
        import statsmodels.api as sm
        Y = RegData_final["PD"]
        X = sm.add_constant(RegData_final[MEVlist[segment2[s]]])
        mod = sm.Logit(Y,X)
        res = mod.fit()
        #print(res.summary())

        # Generate PD forecast
        predict = res.predict(sm.add_constant(macro_scenario[MEVlist[segment2[s]]]))
        PD_Forecast_Qtrs = pd.concat([macro_scenario["period"].loc[0:12], predict.loc[0:12]], axis=1)
        PD_Forecast_Qtrs.columns = ["period", "PD_fcst"]

        # Calculate the shift factor
        PD_hist_mean = RegData_final.PD.mean()

        PD_Shift = PD_Forecast_Qtrs.copy()
        PD_Shift["PD_mean"] = PD_hist_mean
        PD_Shift["fcst"] = PD_Shift["PD_mean"].apply(norm.ppf) - PD_Shift["PD_fcst"].apply(norm.ppf)

        Shiftlist[segment2[s]] = PD_Shift

    
    return Shiftlist, PD_Forecast_Qtrs


# In[77]:


# Generate forecasted PIT matrices by applying the shift factor on the TTC matrices
def pit_mat(pd_fcst, ttc_mat, states, seg, shift):
    #--Obtain PIT Transition Matrices from TTC Matrices by Applying Shift Factors to the Adjusted TTC Matrix--#
    PIT_mat_fcst = {}
    PIT_mat_fcst_AllSeg = {}

    for h in range(len(pd_fcst["period"])):
        PIT_mat_fcst_AllSeg[h] = np.zeros((states * len(seg), states), dtype = float)

    # Define the function to convert TTC matrices to PIT matrices
    def PIT_matrix(TTC_mat, shift_table1, shift_table2, cut, quarter):
        PIT_mat = TTC_mat.copy()
        for i in range(states-1):
            shift_table = shift_table1
            if i>cut:
                shift_table = shift_table2
            #Last Column of Transition Matrix (i.e. PD) Decreased by the Shift Factor#
            PIT_mat.iloc[[i], [states-1]] = norm.cdf(norm.ppf(TTC_mat.iloc[[i],[states-1]]) - shift_table["fcst"][quarter])

            #Adjusting Remaining Columns of Matrix up to the 2nd Column#
            for m in range(2, states):
                start_temp = states - m
                if abs(TTC_mat.iloc[[i], start_temp:states].sum(axis=1)-1).values<0.000000001:
                    PIT_mat.iloc[[i], start_temp] = 1-PIT_mat.iloc[[i], start_temp+1:states].sum(axis=1)
                else:
                    PIT_mat.iloc[[i], start_temp] = norm.cdf(norm.ppf(TTC_mat.iloc[[i], start_temp:states].sum(axis=1)) - shift_table["fcst"][quarter]) - norm.cdf(norm.ppf(TTC_mat.iloc[[i], start_temp+1:states].sum(axis=1)) - shift_table["fcst"][quarter])

            #First Column of Matrix Defined to Ensure Summation of Row to 1#
            PIT_mat.iloc[[i], 0] = 1 - PIT_mat.iloc[[i], 1:states].sum(axis=1)

        return PIT_mat

    #Call Function each segment and Store PIT Matrices in Lists#
    for s in range(len(seg)):
        PIT_mat_fcst_list = []
        for h in range(len(pd_fcst["period"])):
            if seg[s] in ["Fin_US", "Oil_US"]:      
                PIT_mat_fcst_list.append(PIT_matrix(TTC_mat = ttc_mat[s], shift_table1 = shift[seg[s]],                                                   shift_table2 = shift[seg[s]], cut=21, quarter=h))
            elif seg[s] == "Cyc_US_2clst":
                PIT_mat_fcst_list.append(PIT_matrix(TTC_mat = ttc_mat[s], shift_table1 = shift["Cyc_US1"],                                                   shift_table2 = shift["Cyc_US2"], cut=18, quarter=h))
            else:
                PIT_mat_fcst_list.append(PIT_matrix(TTC_mat = ttc_mat[s], shift_table1 = shift["NCy_US1"],                                                   shift_table2 = shift["NCy_US2"], cut=18, quarter=h))
            PIT_mat_fcst_AllSeg[h][s*states:(s+1)*states,:] = PIT_mat_fcst_list[h]
        PIT_mat_fcst[s] = PIT_mat_fcst_list
    
    return PIT_mat_fcst


# In[78]:


# Generate the marginal PD forecast using the forecasted PIT transition matrices
def mpd_fcst(pit_mat, ttc_mat, states, horizon, seg, rs_len, reversion):
    #--Store Marginal PD from PIT Matrices in One Table and Power Cumulative Matrix till Maximum Horizon --#

    PIT_pd_fcst_AllSeg = np.zeros((states*len(seg), horizon), dtype=float)
    for s in range(len(seg)):
        #Cumulative Transition Matrix Till Period within reasonable and supportable period and Storing Marginal PDs#
        PIT_pd_fcst = np.zeros((states, horizon), dtype=float)
        PIT_Cumulative_fcst = pit_mat[s][0].values
        PIT_pd_fcst[:, 0] = PIT_Cumulative_fcst[:,states-1]

        for h in range(1, rs_len):
            mat_cum_temp_prev = PIT_Cumulative_fcst.copy()
            PIT_Cumulative_fcst = np.dot(PIT_Cumulative_fcst, pit_mat[s][h])
            PIT_pd_fcst[:, h] = PIT_Cumulative_fcst[:, states-1] - mat_cum_temp_prev[:,states-1]

        #Use Cumulative TM and Multiply it Further to Obtain Marginal PDs in Future Horizons  
        def Marg_PD(cum_mat, last_mat, margPD_table):
            mat_cum_temp = cum_mat.copy()
            mpd = margPD_table.copy()
            for h in range(rs_len, horizon):
                mat_cum_temp_prev = mat_cum_temp.copy()
                mat_cum_temp = np.dot(mat_cum_temp, last_mat)
                mpd[:, h] = mat_cum_temp[:, states-1] - mat_cum_temp_prev[:, states-1]
            return mpd

        # Two options for the transition matrix used during the mean reversion period
        # Option1: Use TTC matrix
        # Option2: Use last PIT matrix
        if reversion == 'ttc':
            reversion_mat = ttc_mat[s]
        elif reversion == 'pit':
            reversion_mat = pit_mat[s][rs_len-1]

        PIT_pd_fcst_f = Marg_PD(cum_mat = PIT_Cumulative_fcst, last_mat = reversion_mat, margPD_table = PIT_pd_fcst)
        PIT_pd_fcst_AllSeg[s*states:(s+1)*states, :] = PIT_pd_fcst_f

    return PIT_pd_fcst_AllSeg   


# In[79]:


# Generate ECL forecast using the predicted marginal PD and forecast of other risk parameters
def ecl_fcst(data, mpd, states, horizon, seg, rs_len, LGD_option, LGD_val, PP_option, PP_val, EAD_option):

    # Calculate provisions
    Data_temp1 = data.copy()

    #--Define Columns Necessary for CECL Calculation --#
    #CECL: Expected Maturity#
    Data_temp1["CECL_RemMat"] =  Data_temp1["RemMat"]
    #CECL: Exposure#
    Data_temp1["Exp"] = Data_temp1["EAD"]

    #--Initial Number of Columns + 1--#
    col_initial = Data_temp1.shape[1] + 1

    #--Populate with Empty CECL Coloumns till Maximum Horizon --#
    for i in range(1,horizon):
        Data_temp1["CECL_ECL_"+str(i)] = 0


    #--a) Create Maginal PD until Maximum Horizon --#
    Data_MPD = Data_temp1.copy()
    Data_MPD["obs"] = Data_MPD.index+1
    Data_MPD = Data_MPD[["Rating", "obs", "Segment"]]

    MPD = pd.DataFrame(mpd)
    for i in range(0, horizon):
        MPD.rename(columns={i: "MPD_"+str(i+1)}, inplace=True)

    MPD["Rating"] = np.tile(np.arange(1,states+1), len(seg))
    MPD["Segment"] = np.repeat(seg, states)

    Data_MPD = pd.merge(Data_MPD, MPD, how="left", on=["Rating", "Segment"])

    #--b) Create Prepayment Adjusted Marginal PD till Maximum Horizon --#
    Data_AdjMPD = Data_temp1.copy()
    Data_AdjMPD["obs"] = Data_AdjMPD.index+1
    Data_AdjMPD = Data_AdjMPD[["Rating", "obs", "Prepayment", "CECL_RemMat", "Segment", "Fee_Ind", "FAS114_ind"]]

    #--Set prepayment based on prepayment rate from data or user specified prepayment rate--#
    if PP_option == 'pp_user':
        Data_AdjMPD["pp"] = PP_val
    elif PP_option == 'pp_data':
        Data_AdjMPD["pp"] = Data_AdjMPD["Prepayment"]

    for i in range(0, horizon):
        Data_AdjMPD["adj_mpd_"+str(i+1)] = 0


    cum_pd1_prev = Data_MPD["MPD_1"]
    cond_pp_prev = Data_AdjMPD["pp"]
    Data_AdjMPD["adj_mpd_1"] = Data_MPD["MPD_1"]
    cum_pd2_prev = Data_MPD["MPD_1"]
    cum_pp_prev  = Data_AdjMPD["pp"]
    sur_prev     = np.repeat(1, len(cond_pp_prev)) - cum_pd2_prev - cum_pp_prev
    
    for i in range(1, horizon):
        cum_pd1_curr = cum_pd1_prev + Data_MPD["MPD_"+str(i+1)] # current period cum PD based on unadjusted PD
        cond_pd_curr = Data_MPD["MPD_"+str(i+1)] / (1 - cum_pd1_prev) # current period conditional PD
        cond_pp_curr = Data_AdjMPD["pp"] # current period conditional prepayment rate
        Data_AdjMPD["adj_mpd_"+str(i+1)] = cond_pd_curr * sur_prev # current period unconditional PD adjusted for prepay and survive
        adj_mpp_curr = cond_pp_curr * sur_prev # current period unconditional prepayment rate adjusted for default and survive
        cum_pd2_curr = cum_pd2_prev + Data_AdjMPD["adj_mpd_"+str(i+1)] # current period cum PD based on adjusted unconditional PD
        cum_pp_curr  = cum_pp_prev + adj_mpp_curr # current period cumulative prepayment rate based on adjusted unconditional prepayment rate
        sur_curr     = 1 - cum_pd2_curr - cum_pp_curr # current period surviorship
        # Assign current period values as perviou period values in the next period calculation
        cum_pd1_prev = cum_pd1_curr.copy()
        sur_prev     = sur_curr.copy()
        cum_pd2_prev = cum_pd2_curr.copy()
        cum_pp_prev  = cum_pp_curr.copy()

    #For certain outstanding (deferred fees, uneanred discount, etc), use marginal PD without prepayment impact
    for i in range(1, horizon+1):
        Data_AdjMPD.loc[Data_AdjMPD.Fee_Ind==1, "adj_mpd_"+str(i)] = Data_MPD.loc[Data_AdjMPD.Fee_Ind==1, "MPD_"+str(i)]

    #Set marginal PD of the impaired loans (FAS 114) to 1 in the first quarter and to 0 beyond the first quarter #
    Data_AdjMPD.loc[Data_AdjMPD.FAS114_ind==1, "adj_mpd_1"] = 1
    for i in range(2, horizon+1):
        Data_AdjMPD.loc[Data_AdjMPD.FAS114_ind==1, "adj_mpd_"+str(i)] = 0

    #--c) Create LGD till Maximum Horizon --#

    Data_LGD = Data_temp1.copy()
    Data_LGD["obs"] = Data_LGD.index+1
    Data_LGD = Data_LGD[["Rating", "obs", "LGD", "CECL_RemMat", "Segment"]]

    if LGD_option == 'lgd_data':
        for i in range(1, horizon+1):
            Data_LGD["lgd_"+str(i)] = Data_LGD['LGD']
    elif LGD_option == 'lgd_user':
        for i in range(1, horizon+1):
            Data_LGD["lgd_"+str(i)] = LGD_val
    elif LGD_option == 'lgd_fj':
        Data_LGD["FJ_K"] = (norm.ppf(Data_AdjMPD["adj_mpd_1"].mean()) - norm.ppf(Data_AdjMPD["adj_mpd_1"].mean()*Data_LGD["LGD"].mean()))/(1 - 0.24)**0.5
        for i in range(1, horizon+1):
            Data_LGD["lgd_"+str(i)] = norm.cdf(norm.ppf(Data_AdjMPD["adj_mpd_"+str(i)]) - Data_LGD["FJ_K"])/Data_AdjMPD["adj_mpd_"+str(i)]

    #--d) Create EAD till Maximum Horizon --#
    Data_EAD = Data_temp1.copy()
    Data_EAD["obs"] = Data_EAD.index+1
    Data_EAD = Data_EAD[["Rating", "obs", "Exp", "CECL_RemMat", "Segment", "Amort_Type", "EIR"]]

    for i in range(0, horizon):
        Data_EAD["ead_"+str(i+1)] = 0

    #--Set EAD based on Amortization Type--#
    cond1 = (Data_EAD.Amort_Type=="IO") | (Data_EAD.Amort_Type=="Irregular") | ((Data_EAD.Amort_Type=="Level") & (Data_EAD.EIR==0))
    cond2 = Data_EAD.Amort_Type=="StraightLine"
    cond3 = (Data_EAD.Amort_Type=="Level") & (Data_EAD.EIR>0)
    if EAD_option == "on":
        for i in range(1, horizon+1):
            Data_EAD.loc[cond1, "ead_"+str(i)] = Data_EAD.loc[cond1, "Exp"]
            Data_EAD.loc[cond2, "ead_"+str(i)] = Data_EAD.loc[cond2, "Exp"] * (1 - ((i-1)/Data_EAD["CECL_RemMat"]))
            Data_EAD.loc[cond3, "ead_"+str(i)] = Data_EAD.loc[cond3, "Exp"] * ((1+Data_EAD.loc[cond3, "EIR"]/4)**Data_EAD["CECL_RemMat"] - (1+Data_EAD.loc[cond3, "EIR"]/4)**(i-1)) / ((1+Data_EAD.loc[cond3, "EIR"]/4)**Data_EAD["CECL_RemMat"] - 1)
    elif EAD_option == "off":
        for i in range(1, horizon+1):
            Data_EAD["ead_"+str(i)] = Data_EAD.Exp


    #--e) Calculate ECL till Maximum Horizon --#
    for i in range(1, horizon+1):
        cond1 = Data_temp1.CECL_RemMat < i-1
        cond2 = (Data_temp1.CECL_RemMat >= i-1) & (Data_temp1.CECL_RemMat < i)
        cond3 = Data_temp1.CECL_RemMat >= i

        Data_temp1.loc[cond1, "CECL_ECL_"+str(i)] = 0
        Data_temp1.loc[cond2, "CECL_ECL_"+str(i)] = Data_AdjMPD.loc[cond2, "adj_mpd_"+str(i)] * Data_LGD.loc[cond2, "lgd_"+str(i)] * Data_EAD.loc[cond2, "ead_"+str(i)] * (Data_temp1.loc[cond2, "CECL_RemMat"] - (i-1))
        Data_temp1.loc[cond3, "CECL_ECL_"+str(i)] = Data_AdjMPD.loc[cond3, "adj_mpd_"+str(i)] * Data_LGD.loc[cond3, "lgd_"+str(i)] * Data_EAD.loc[cond3, "ead_"+str(i)]

    col_final = Data_temp1.shape[1]
    Data_temp1["ECL_Total"] = Data_temp1.iloc[:,col_initial-1:col_final].sum(axis=1)
    
    return Data_temp1, Data_AdjMPD, Data_LGD


# In[86]:
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'xlsx' in filename:
        df = pd.read_excel(io.BytesIO(decoded))
    else:
        df = pd.DataFrame()
        
    return df
# In[87]:


def ECL_Calc(portfolio, scenario, reversion, rs_len, LGD_option, LGD_val, PP_option, PP_val, EAD_option):
    
    account_data = portfolio

    # Import segment level TTC transition matrix and historical default rate
    segment = ["Fin_US","Oil_US","Cyc_US_2clst","NCy_US"]
    TMlist = {}
    DRlist = {}

    for s in segment:
        TMlist[s] = pd.read_excel("TM_AllSeg.xls", sheet_name = s)
        DRlist[s] = pd.read_csv("DR_"+s+".csv")
        
    # Generate the number of ratings and number of forecast period based on TTC transition matrix 
    # and the maximum remaining maturity of the portfolio
    ttc_mat = rel_mat(segment, TMlist) # Normallize the rating migration matrix for each segment so that each row adds up to 1
    states = ttc_mat[0].shape[1]
    horizon = max(20, math.ceil(account_data["RemMat"].max() + 0.0001))
    
    # Import historical macroeconomic variables and scenario forecast
    macro_hist = pd.read_csv("Supervisory historical Domestic.csv")
    #macro_base = pd.read_csv("base.csv")
    #macro_adv = pd.read_csv("adv.csv")
    #macro_sa = pd.read_csv("sa.csv")

    #macro_scenario_base = macro_hist.append(macro_base)
    #macro_scenario_adv = macro_hist.append(macro_adv)
    #macro_scenario_sa = macro_hist.append(macro_sa)
    macro_scenario = macro_hist.append(scenario)
    
    # Keep the MEVs used in regression and generate lags up to 2 quarters
    macro_hist_final = MacroTrans(macro_hist)
    #if scenario == 'ba':
    #    macro_scenario_final = MacroTrans(macro_scenario_base)
    #elif scenario == 'ad':
    #    macro_scenario_final = MacroTrans(macro_scenario_adv)
    #elif scenario == 'sa':
    #    macro_scenario_final = MacroTrans(macro_scenario_sa)
        
    macro_scenario_final = MacroTrans(macro_scenario)

    # Generate the shift factors for all segments in the forecast horizon
    Shiftlist, PD_Forecast_Qtrs = shift_list(DRlist, macro_hist_final, macro_scenario_final)
    
    # Generate forecasted PIT matrices by applying the shift factor on the TTC matrices
    PIT_mat_fcst = pit_mat(PD_Forecast_Qtrs, ttc_mat, states, segment, Shiftlist)
    
    # Generate the marginal PD forecast using the forecasted PIT transition matrices
    Marginal_PD = mpd_fcst(PIT_mat_fcst, ttc_mat, states, horizon, segment, rs_len, reversion)
    
    # Generate ECL forecast using the predicted marginal PD and forecast of other risk parameters
    ECL_loan, ECL_PD, ECL_LGD = ecl_fcst(account_data, Marginal_PD, states, horizon, segment, rs_len, LGD_option, LGD_val, PP_option, PP_val, EAD_option)

    return ECL_loan, ECL_PD, ECL_LGD



# In[ ]:

def port_plot(data):
    data['Industry'] = ''
    data.loc[data.Segment=='Fin_US', 'Industry'] = 'Financial Instituions'
    data.loc[data.Segment=='Oil_US', 'Industry'] = 'Oil & Gas'
    data.loc[data.Segment=='Cyc_US_2clst', 'Industry'] = 'Cyclical'
    data.loc[data.Segment=='NCy_US', 'Industry'] = 'Non-Cyclical'
    
    bal_ind    = data[['Bal', 'Industry']].groupby(['Industry']).sum()/1000000
    bal_rating = data[['Bal', 'CurRR']].groupby(['CurRR']).sum()/1000000
    
    fig = tools.make_subplots(rows=2, cols=1,specs=
                                  [[{}],
                                  [{}]]
                             )
    fig.add_bar(x=bal_ind.index, y=bal_ind['Bal'], row=1,col=1)
    fig.add_bar(x=bal_rating.index, y=bal_rating['Bal'], row=2,col=1)
                
    fig['layout'].update(height = 700, margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False
        })
    
    fig['layout']['xaxis1'].update(title='Industry')
    fig['layout']['xaxis2'].update(title='Rating', tickmode='linear')
    
    fig.layout.template = 'plotly_dark'
                
    return fig

def sc_plot(sc):
    macro_hist = pd.read_csv("Supervisory historical Domestic.csv")
    macro_scenario = macro_hist[macro_hist.Date > "2017 Q4"].append(sc)
    macro_scenario['spread'] = macro_scenario['BBB corporate yield'] - macro_scenario['10-year Treasury yield']
        
    fig = tools.make_subplots(rows=2, cols=2,specs=
                                  [[{}, {}],
                                   [{}, {}]]
                             )
    fig.add_scatter(x=macro_scenario['Date'],y=macro_scenario['Real GDP growth'],row=1,col=1,line=dict(color='#188CE5'))
    fig.add_scatter(x=macro_scenario['Date'],y=macro_scenario['spread'],row=1,col=2,line=dict(color='#FF4136'))
    fig.add_scatter(x=macro_scenario['Date'],y=macro_scenario['Market Volatility Index (Level)'],row=2,col=1,line=dict(color='#2DB757'))
    fig.add_scatter(x=macro_scenario['Date'],y=macro_scenario['WTI Price'],row=2,col=2,line=dict(color='#27ACAA'))
                
    fig['layout'].update(height=700, margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False,
        })
    
    fig['layout']['yaxis1'].update(title='GDP Growth')
    fig['layout']['yaxis2'].update(title='BBB Spread')
    fig['layout']['yaxis3'].update(title='VIX Index')
    fig['layout']['yaxis4'].update(title='WTI Price')
    
    fig.layout.template = 'plotly_dark'
                
    return fig

#def ecl_plot(data):
#    horizon = max(20, math.ceil(data["RemMat"].max() + 0.0001))
#    ECL_qtr = data[data.columns[-(1+horizon):-1]].sum(axis=0)
#    yr = math.ceil(horizon/4)
#    ECL_yr = np.zeros(yr+1)
#    for i in range(0, yr):
#        ECL_yr[i] = ECL_qtr[i*4:(i+1)*4].sum()/1000000
#    ECL_yr[yr] = ECL_yr[0:yr].sum()
#
#    ECL_plot = pd.DataFrame(ECL_yr)
#    new_index = []
#    for i in range(1, yr+1):
#        new_index.append('Yr_'+str(i))
#    new_index.append('Total') 
#    ECL_plot['new_index'] = new_index
#    ECL_plot.set_index('new_index', inplace=True)
#
#    trace = go.Scatter(
#        x = ECL_plot.index, 
#        y = ECL_plot[ECL_plot.columns[0]],
#        text = [round(x, 2) for x in ECL_plot[ECL_plot.columns[0]].tolist()],
#        textposition='auto',
#        marker=dict(
#            color=['rgba(0, 157, 255, 1)', 'rgba(0, 157, 255, 1)',
#                   'rgba(0, 157, 255, 1)', 'rgba(0, 157, 255, 1)',
#                   'rgba(0, 157, 255, 1)', 'rgba(0, 157, 255, 1)',
#                   'rgba(255, 234, 5, 1)']),
#    )
#
#    data = [trace]
#    layout = go.Layout(
#        title = 'Annual ECL ($mm)'
#    )
#
#    fig = go.Figure(data=data, layout = layout)
#    
#    fig['layout'].update(height=600,width=800,margin={'t':0})
#    fig['layout'].update({
#         'plot_bgcolor':'rgba(0,0,0,0)',
#         'paper_bgcolor':'rgba(0,0,0,0)',
#         'showlegend':False
#         })
#    fig['layout']['yaxis1'].update(title=None)
#    fig['layout']['xaxis1'].update(title=None)
#    
#    fig.layout.template = 'plotly_dark'
#    
#    return fig
    
def ecl_plot(data):
    horizon = max(20, math.ceil(data["RemMat"].max() + 0.0001))
    
    plot_data = data.iloc[:,data.shape[1]-horizon-1:data.shape[1]-2].sum()/1000
    plot_data = plot_data.reset_index()
    plot_data.columns = ['index', 'ecl']
    
    fig = go.Figure()
    
    fig.add_scatter(x=plot_data.index+1,y=plot_data['ecl'],line={'color':'#2DB757'})
                    
    fig['layout'].update(height=600, margin={'t':0})
    fig['layout'].update({
         'plot_bgcolor':'rgba(0,0,0,0)',
         'paper_bgcolor':'rgba(0,0,0,0)',
         'showlegend':False
         })
    fig['layout']['yaxis1'].update(title=None)
    fig['layout']['xaxis1'].update(title=None, tickmode='linear')
    
    fig.layout.template = 'plotly_dark'
    
    return fig

def pd_lgd_plot(data1, data2, group):
    
    horizon = max(20, math.ceil(data1["CECL_RemMat"].max() + 0.0001))
    if group == 'port':
        pd_data = data1
        lgd_data = data2
    elif group == 'ig':
        pd_data = data1[data1.Rating<=10]
        lgd_data = data2[data2.Rating<=10]
    else:
        for r in range(11,22):
            if group == str(r):
                pd_data = data1[data1.Rating==r]
                lgd_data = data2[data2.Rating==r]
    
    pd_plot = pd_data.iloc[:,pd_data.shape[1]-horizon:pd_data.shape[1]-1].mean()
    pd_plot = pd_plot.reset_index()
    pd_plot.columns = ['index', 'pd']
    lgd_plot = lgd_data.iloc[:,lgd_data.shape[1]-horizon:lgd_data.shape[1]-1].mean()
    lgd_plot = lgd_plot.reset_index()
    lgd_plot.columns = ['index', 'lgd']
    
    plot_data = pd.concat([pd_plot['pd'], lgd_plot['lgd']], axis=1)

    fig = tools.make_subplots(rows=2, cols=1,specs=
                                  [[{}],
                                  [{}]]
                             )
    fig.add_scatter(x=plot_data.index+1, y=plot_data['pd'],row=1,col=1,line=dict(color='#188CE5'))
    fig.add_scatter(x=plot_data.index+1, y=plot_data['lgd'],row=2,col=1,line=dict(color='#FF4136'))
                
    fig['layout'].update(height = 700, margin={'t':0})
    fig['layout'].update({
        'plot_bgcolor':'rgba(0,0,0,0)',
        'paper_bgcolor':'rgba(0,0,0,0)',
        'showlegend':False
        })
    
    fig['layout']['yaxis1'].update(title='PD')
    fig['layout']['yaxis2'].update(title='LGD')
    fig['layout']['xaxis1'].update(title='Forecast Quarter', tickmode='linear')
    fig['layout']['xaxis2'].update(title='Forecast Quarter', tickmode='linear')
    
    fig.layout.template = 'plotly_dark'
                
    return fig