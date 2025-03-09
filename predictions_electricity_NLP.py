#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:50:47 2022

@author: yunbai
"""

##################### imports #####################
import pandas as pd
import numpy as np
import pickle
import datetime
from datetime import timedelta
import time
import os
import auxiliary_evaluation as auxEval
import auxiliary_configuration as auxConfig
from auxiliary_generic import *
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression,LassoCV,RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import RegressorChain

import warnings
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR
from prophet import Prophet
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from IPython.core.pylabtools import figsize
import matplotlib.pyplot as plt
import pylab as py
import pickle

import lime
import lime.lime_tabular

#%%
def logError(e):
    try:
        print(e)
    except Exception as e:
        logError(e)

## Split data
def getTrainTest(X,Y,configuration):
    X_train, X_valid, X_test = X[X.index<='2018-06-01'],X[(X.index>'2018-06-01')&(X.index<='2020-06-01')], X[X.index>'2020-06-01']
    Y_train, Y_valid, Y_test = Y[Y.index<='2018-06-01'],Y[(Y.index>'2018-06-01')&(Y.index<='2020-06-01')], Y[Y.index>'2020-06-01']
    
    Xcolumns,Ycolumns = X.columns,Y.columns
    scaler_x = StandardScaler()
    scaler_x.fit(X_train)
    scaler_y = StandardScaler()
    scaler_y.fit(Y_train)
    
    X_train = scale_data(Xcolumns,scaler_x,X_train)
    X_valid = scale_data(Xcolumns,scaler_x,X_valid)
    X_test = scale_data(Xcolumns,scaler_x,X_test)
    
    Y_train = scale_data(Ycolumns,scaler_y,Y_train)
    Y_valid = scale_data(Ycolumns,scaler_y,Y_valid)
    Y_test = scale_data(Ycolumns,scaler_y,Y_test)
   
    return X_train,X_valid,X_test,Y_train,Y_valid,Y_test,scaler_x,scaler_y

## Granger test
def GrangerGettingFeats(Ygranger,TextCols,Ycol,lags):   
    Load = Ygranger[Ycol].mean(axis=1).to_frame()
    Load = Load - Load.shift(1)
    Load = Load.dropna()
    
    selectedFeatures = []
    for itemX in TextCols:
        ## ADF Null hypothesis: there is a unit root, meaning series is non-stationary
        Text_X = Ygranger[[itemX]]
        adfResult = adfuller(Text_X)
        if adfResult[1] > 0.05:
            print('*'*50)
            print(itemX)
            print('*'*50)
            Text_X = Text_X - Text_X.shift(1)
            Text_X = Text_X.dropna()
        
        X2Y = Load.merge(Text_X,left_on='Date', right_on='Date', how='inner')
        Y2X = Text_X.merge(Load,left_on='Date', right_on='Date', how='inner')
                
        try:
            GrangerX2Y = grangercausalitytests(X2Y, maxlag=[30])           
            GrangerY2X = grangercausalitytests(Y2X, maxlag=[30])
            keyX2Y = list(GrangerX2Y.keys())[0]
            keyY2X = list(GrangerY2X.keys())[0]
                    
            if GrangerX2Y[keyX2Y][0]['ssr_ftest'][1]<=0.05 and GrangerY2X[keyY2X][0]['ssr_ftest'][1]>0.05:
                selectedFeatures.append(itemX)  
        except:
            pass
                
    return selectedFeatures

## Forecast functions 
def CV_GS_valid(X_train,X_valid,X_test,Y_train,Y_valid,Y_test,scaler_y):
    # SVR model
    X_train_S2 = pd.concat([X_train,X_valid])
    Y_train_S2 = pd.concat([Y_train,Y_valid])
    
    t1 = time.time()
    estSVR = SVR()
    param_grid_SVR = [{'estimator__base_estimator__C':[0.1,1,10,100],
                       'estimator__base_estimator__kernel':['linear','rbf','sigmoid']}]
    pipeline = Pipeline(steps = [('estimator', RegressorChain(estSVR))])     
    clfSVR = GridSearchCV(pipeline, param_grid_SVR, cv=5,n_jobs=-1)
    clfSVR.fit(X_train_S2,Y_train_S2)
    t2 = time.time()
    print('SVR CV and GS training time(s)',t2-t1)
    best_SVR = clfSVR.best_estimator_.steps[0][1]
    
    best_SVR = RegressorChain(SVR(C=0.1, kernel='linear'))
    best_SVR.fit(X_train_S2,Y_train_S2)
    
    # MLP model
    t1 = time.time()
    estMLP = MLPRegressor(random_state=1,early_stopping=True)
    param_grid_MLP = [{'hidden_layer_sizes':[(512,256,128,64,),(256,128,64,),(128,64,),(64,)],
                       'activation':['identity', 'logistic', 'tanh', 'relu']}]
    clfMLP = GridSearchCV(estMLP, param_grid_MLP, cv=5,n_jobs=-1)
    clfMLP.fit(X_train_S2,Y_train_S2)
    t2 = time.time()
    print('MLP CV and GS training time(s)',t2-t1)
    best_MLP = clfMLP.best_estimator_ #(256, 128, 64) tanh
    
    #ExtraTrees model
    t1 = time.time()
    estETR = ExtraTreesRegressor()
    param_grid_ETR = [{'n_estimators': [50, 100, 500, 1000],
                       'bootstrap': [False, True]}]
    clfETR = GridSearchCV(estETR, param_grid_ETR, cv=5,n_jobs=-1)
    clfETR.fit(X_train_S2,Y_train_S2)
    t2 = time.time()
    print('ETR CV and GS training time(s)',t2-t1)
    best_ETR = clfETR.best_estimator_   
    
    best_ETR = ExtraTreesRegressor(n_estimators = 1000)
    best_ETR.fit(X_train_S2,Y_train_S2)
    
    fore_truth_SVR = ExtraTreePre(best_SVR,X_test,Y_test,scaler_y)
    fore_truth_MLP = ExtraTreePre(best_MLP,X_test,Y_test,scaler_y)
    fore_truth_ETR = ExtraTreePre(best_ETR,X_test,Y_test,scaler_y)
    
    ForecastError = calculate_error(fore_truth_ETR[['Truth','Forecasts']],'Truth','Forecasts')
    ForecastError = ForecastError[ForecastError.index>datetime.date(2020,6,2)]
    ForecastError = ForecastError[ForecastError.index<datetime.date(2021,4,13)]
            
    return best_SVR,best_MLP,best_ETR

def tuningCV_ExtraTrees(X, Y, configuration):
    if configuration['crossvalidation'] == 'yes':
        
        estimator = ExtraTreesRegressor(n_estimators=int(configuration['n_estimators']))
        clf = GridSearchCV(estimator=estimator, param_grid=configuration['tuning_parameters'], 
                           scoring=configuration['score'],n_jobs=-3)
        clf.fit(X, Y)
        bestEstimator = clf.best_estimator_
        print('#########')
        print(clf.best_params_)
    else:
        estimator = ExtraTreesRegressor(n_estimators = 1000,
                                        max_features = 'auto',
                                        bootstrap = False,
                                        n_jobs = -3)
                                        # random_state = 111)
        bestEstimator = estimator.fit(X, Y)
    return bestEstimator

def ExtraTreePre(model,X_input,Y_input,scaler_y):
    Y_pres = model.predict(X_input)
    Y_pres = scaler_y.inverse_transform(Y_pres)
    Y_pres_df = pd.DataFrame(Y_pres,columns=Y_input.columns,index=Y_input.index)
                
    Y_true = Y_input.copy()
    Y_input = scaler_y.inverse_transform(Y_input)
    Y_input = pd.DataFrame(Y_input,columns=Y_true.columns,index=Y_true.index)
                                
    Y_pres_df = expand_data(Y_pres_df)
    Y_pres_df.columns = ['Forecasts']
    Y_input = expand_data(Y_input)
    Y_input.columns = ['Truth']  
    fore_truth = pd.concat([Y_input,Y_pres_df],axis=1)
    
    return fore_truth

## Experiments
# Compare models with holidays/weekdays/temperature
def trainModels_for_selecting_on_test(configuration,lags,YXtableDict,FeatDict):
    
    try:
        tableList = ['YXtable_NoText','YXtable_holiday', 'YXtable_weekend',
                     'YXtable_temperature','YXtable_week_temp','YXtable_week_hol_temp']
        selectResults = dict()
        for tableName in tableList:
            print(tableName)
            X = YXtableDict[tableName+'_X']
            Y = YXtableDict[tableName+'_Y']            
            X_train,X_valid,X_test,Y_train,Y_valid,Y_test,scaler_x,scaler_y= getTrainTest(X,Y,configuration)
            X_train_S2 = pd.concat([X_train,X_valid])
            Y_train_S2 = pd.concat([Y_train,Y_valid])           
            
            estimatorExtraTrees = tuningCV_ExtraTrees(X_train_S2, Y_train_S2, configuration)
            
            Y_pres_test = estimatorExtraTrees.predict(X_test)
            Y_pres = scaler_y.inverse_transform(Y_pres_test)
            Y_pres_df = pd.DataFrame(Y_pres,columns=Y_test.columns,index=Y_test.index)
                
            Y_true = Y_test.copy()
            Y_test = scaler_y.inverse_transform(Y_test)
            Y_test = pd.DataFrame(Y_test,columns=Y_true.columns,index=Y_true.index)
            
            Y_pres_df = expand_data(Y_pres_df)
            Y_pres_df.columns = ['Forecasts']
            Y_test = expand_data(Y_test)
            Y_test.columns = ['Truth']  
            fore_truth = pd.concat([Y_test,Y_pres_df],axis=1)
            
            ForecastError = calculate_error(fore_truth[['Truth','Forecasts']],'Truth','Forecasts')
            ForecastError = ForecastError[ForecastError.index>datetime.date(2020,6,2)]
            ForecastError = ForecastError[ForecastError.index<datetime.date(2021,4,13)]
            
            if tableName=='YXtable_holiday':
                fore_truth = fore_truth.merge(YXtableDict['benchmark'],left_index=True,right_index=True,how='left')
                benchmarkError = calculate_error(fore_truth[['Truth','ForecastedLoad']],'Truth','ForecastedLoad')
                benchmarkError = benchmarkError[benchmarkError.index>datetime.date(2020,6,2)]
                benchmarkError = benchmarkError[benchmarkError.index<datetime.date(2021,4,13)]
                selectResults['benchmark'] = [benchmarkError.rmse.mean(),benchmarkError.mae.mean(),benchmarkError.smape.mean()]

            selectResults[tableName] = [ForecastError.rmse.mean(),ForecastError.mae.mean(),ForecastError.smape.mean()]

        return selectResults
    
    except Exception as e:
        logError(e)

# Get textual features for model with weekdays and temperatures
def getNLP_for_week_tem(YXtableDict,configuration,lags,electric=False,section=None):
    """
    This function is to find the related text features selected by Granger test,
    if electric = True, then find the electric-related news features
    """
    tableName = 'YXtable_week_temp'
    X = YXtableDict[tableName+'_X']
    Y = YXtableDict[tableName+'_Y']
    Xcol,Ycol = list(X.columns),list(Y.columns)
    X_Y = pd.concat([X,Y],axis=1)
    X_Y = treatNAN(X_Y, configuration)
    
    week_tem_NLP_dict = dict()
    week_tem_NLP_dict[tableName+'_X'] = X_Y[Xcol]
    week_tem_NLP_dict[tableName+'_Y'] = X_Y[Ycol]
    
    
    requestDict = {'periodString': 'max',
                   'db_connector': configuration['db_connector_address']
                      }
    
    selectedFeat = dict()
    
    # if electric == True:
    #     elecRoot = ''
    #     configuration['NLP_features_table'] = [elecRoot+'electric-count-feat.csv',
    #                                            elecRoot+'electric-WordFreqDf.csv',
    #                                            elecRoot+'electric-senti-feat.csv',
    #                                            elecRoot+'electric_topicDistribution_grouped.csv',
    #                                            elecRoot+'electric_features_GloVe_grouped.csv']
    # if section != None:
    #     sectionRoot = ''
    #     configuration['NLP_features_table'] = [sectionRoot+'Count/'+section+'-count-feat.csv',
    #                                            sectionRoot+'WordFreq/'+section+'-wordFreq-feat.csv',
    #                                            sectionRoot+'Sentiments/'+section+'senti-feat.csv',
    #                                            sectionRoot+'Topics/'+section+'topic-feat.csv',
    #                                            sectionRoot+'GloVe/'+section+'glove-feat.csv']
    
    
    for i in range(len(configuration['NLP_features_table'])):
        tableName = configuration['NLP_features_table'][i]
        # if i == 0:
        #     tableName = 'count'
        # elif i == 1:
        #     tableName = 'wordFreq'
        # elif i == 2:
        #     tableName = 'senti'
        # elif i == 3:
        #     tableName = 'topic'
        # else:
        #     tableName = 'GloVe'
            
        if electric == True or section != None:
            df = pd.read_csv(tableName)
        else:    
            requestDict['tableName'] = tableName
            df = readDataFromDB(requestDict)
        df = df.drop_duplicates(subset=['Date'])
        df = resampleDataFrame(df, configuration)
        df = df.loc[:,(df!=df.iloc[0]).any(axis=0)]    # delete same cols    
        
        YXtable_temp = X_Y.merge(df, left_on='Date', right_on='Date', how='left')
        YXtable_temp = treatNAN(YXtable_temp, configuration)        
        
        #Granger Test        
        featureSelected = GrangerGettingFeats(YXtable_temp,list(df.columns),Ycol,lags)
        del df
        selectedFeat[tableName] = featureSelected 
        
        # log transform
        for feat in featureSelected:
            YXtable_temp[feat] = YXtable_temp[feat].apply(lambda x: np.log(x+1))
        week_tem_NLP_dict[tableName+'_Y'] = YXtable_temp[Ycol]
        week_tem_NLP_dict[tableName+'_X'] = YXtable_temp[list(Xcol)+featureSelected]
            
    return week_tem_NLP_dict,selectedFeat

# Compare other word embeddings
def getEmbeddings_for_week_tem(YXtableDict,configuration,lags):
    """
    This function is to find the related word2vec or tfidf features
    """
    tableName = 'YXtable_week_temp'
    X = YXtableDict[tableName+'_X']
    Y = YXtableDict[tableName+'_Y']
    Xcol,Ycol = list(X.columns),list(Y.columns)
    X_Y = pd.concat([X,Y],axis=1)
    X_Y = treatNAN(X_Y, configuration)
    
    embedding_NLP_dict = dict()
    embedding_NLP_dict[tableName+'_X'] = X_Y[Xcol]
    embedding_NLP_dict[tableName+'_Y'] = X_Y[Ycol]
       
    root = './Dataset/OtherEmbeddings/'
    embeddingName = ['groupedTfidf.csv','groupedW2Vdf.csv']
    selectedFeat = dict()
    for table in embeddingName:
        df = pd.read_csv(root+table)
        df = df.drop_duplicates(subset=['Date'])
        df = resampleDataFrame(df, configuration)
        df = df.loc[:,(df!=df.iloc[0]).any(axis=0)]    # delete same cols    
        
        YXtable_temp = X_Y.merge(df, left_on='Date', right_on='Date', how='left')
        YXtable_temp = treatNAN(YXtable_temp, configuration)        
        
        #Granger Test        
        featureSelected = GrangerGettingFeats(YXtable_temp,list(df.columns),Ycol,lags)
        del df
        selectedFeat[table] = featureSelected 
        
        # log transform
        for feat in featureSelected:
            YXtable_temp[feat] = YXtable_temp[feat].apply(lambda x: np.log(x+1))
        embedding_NLP_dict[table+'_Y'] = YXtable_temp[Ycol]
        embedding_NLP_dict[table+'_X'] = YXtable_temp[list(Xcol)+featureSelected]
            
    return embedding_NLP_dict,selectedFeat

# Compare and select the number of lags for Granger test
def GrangerLagsSelect(YXtableDict,configuration,lags,selectedFeat):
    """
    selectedFeat is already known, read from the file.
    This function is to compute the AIC values for the selected text features.
    Set the lags in [1,7,30,90]
    """
    tableName = 'YXtable_week_temp'
    X = YXtableDict[tableName+'_X']
    Y = YXtableDict[tableName+'_Y']
    Xcol,Ycol = list(X.columns),list(Y.columns)
    X_Y = pd.concat([X,Y],axis=1)
    X_Y = treatNAN(X_Y, configuration)
       
    requestDict = {'periodString': 'max',
                   'db_connector': configuration['db_connector_address']
                   }
    
    textTables = ['title_WordFreqDf','body_features_senti',
                  'body_topicDistributionDf_grouped','body_features_Glove_grouped']
    
    GrangerLagDf = pd.DataFrame()
    for table in textTables:
        requestDict['tableName'] = table
        df = auxData.readDataFromDB(requestDict)
        df = df.drop_duplicates(subset=['Date'])
        df = resampleDataFrame(df, configuration)
        df = df.loc[:,(df!=df.iloc[0]).any(axis=0)]    # delete same cols    
        
        YXtable_temp = X_Y.merge(df, left_on='Date', right_on='Date', how='left')
        YXtable_temp = treatNAN(YXtable_temp, configuration)  
        
        feats = selectedFeat[table]
        Load = YXtable_temp[Ycol].mean(axis=1).to_frame()
        # Load = Load - Load.shift(1)
        Load = Load.dropna()
        
        for itemX in feats:
            GrangerLag = dict()
            GrangerLag['featName'] = itemX
            Text_X = YXtable_temp[[itemX]]
            # adfResult = adfuller(Text_X)
            # if adfResult[1] > 0.05:
            #     print('*'*50)
            #     print(itemX)
            #     print('*'*50)
            #     Text_X = Text_X - Text_X.shift(1)
            #     Text_X = Text_X.dropna()
            
            X2Y = Load.merge(Text_X,left_on='Date', right_on='Date', how='inner')
            model = VAR(X2Y)
            for i in [1,7,30,90]:
                result = model.fit(i)
                GrangerLag['AIC-Lag'+str(i)] = result.aic
            GrangerLagDf = GrangerLagDf.append(GrangerLag,ignore_index=True)
        
    return GrangerLagDf

# Get the combined textual features

def getCombineFeat(Week_tem_NLP_dict,selectedFeat):
    CombineFeatDict = dict()
    '''WF: word frequency, SE: sentiment, TD:topic distribution, GW: Glove Word embedding'''
    '''combine the word frequency of titles, des and body together'''
    WfColT = selectedFeat['title_WordFreqDf']
    WfColD = selectedFeat['des_WordFreqDf']
    WfColB = selectedFeat['body_WordFreqDf']
    
    WfDfT_X = Week_tem_NLP_dict['title_WordFreqDf_X'][WfColT]
    WfDfD_X = Week_tem_NLP_dict['des_WordFreqDf_X'][WfColD]
    WfDfB_X = Week_tem_NLP_dict['body_WordFreqDf_X'][WfColB]
    WfDfT_X.columns = [name+'-T' for name in WfDfT_X]
    WfDfD_X.columns = [name+'-D' for name in WfDfD_X]
    WfDfB_X.columns = [name+'-B' for name in WfDfB_X]
    
    Load_X = Week_tem_NLP_dict['des_features_count_X']
    Load_Y = Week_tem_NLP_dict['des_features_count_Y']
    
    Ycols = list(Week_tem_NLP_dict['des_features_count_Y'].columns)
    XcolsTDB = list(Load_X.columns)+list(WfDfT_X.columns)+list(WfDfD_X.columns)+list(WfDfB_X.columns)
    XcolsT = list(Load_X.columns)+list(WfDfT_X.columns)
    Xcols = list(Load_X.columns)
    WfTDB = pd.concat([Load_X,WfDfT_X,WfDfD_X,WfDfB_X,Load_Y],axis=1)
    WfTDB = WfTDB.dropna()
    
    WfTDB_X,WfTDB_Y = WfTDB[XcolsTDB],WfTDB[Ycols]    
    CombineFeatDict['WfTDB_X'],CombineFeatDict['WfTDB_Y'] = WfTDB_X,WfTDB_Y
    
    WfT = pd.concat([Load_X,WfDfT_X,Load_Y],axis=1)
    WfT = WfT.dropna()
    WfT_X = WfT[XcolsT]
    WfT_Y = Week_tem_NLP_dict['title_WordFreqDf_Y']
    
    '''combine the WF from titles with sentiment from body'''
    Senti = Week_tem_NLP_dict['body_features_senti_X'][selectedFeat['body_features_senti']]
    WfT_Senti = pd.concat([WfT_X,Senti,WfT_Y],axis=1)
    WfT_Senti = WfT_Senti.dropna()
    Xcols_comb = XcolsT+list(Senti.columns)
    CombineFeatDict['WfT_Senti_X'],CombineFeatDict['WfT_Senti_Y'] = WfT_Senti[Xcols_comb],WfT_Senti[Ycols]
    
    '''combine the WF from titles with topics from body'''
    Topic = Week_tem_NLP_dict['body_topicDistributionDf_grouped_X'][selectedFeat['body_topicDistributionDf_grouped']]
    WfT_topic = pd.concat([WfT_X,Topic,WfT_Y],axis=1)
    WfT_topic = WfT_topic.dropna()
    Xcols_comb = XcolsT+list(Topic.columns)
    CombineFeatDict['WfT_Topic_X'],CombineFeatDict['WfT_Topic_Y'] = WfT_topic[Xcols_comb],WfT_topic[Ycols]
    
    '''combine the Wf from titles with glove from body'''
    Glove = Week_tem_NLP_dict['body_features_Glove_grouped_X'][selectedFeat['body_features_Glove_grouped']]
    WfT_glove = pd.concat([WfT_X,Glove,WfT_Y],axis=1)
    WfT_glove = WfT_glove.dropna()
    Xcols_comb = XcolsT+list(Glove.columns)
    CombineFeatDict['WfT_Glove_X'],CombineFeatDict['WfT_Glove_Y'] = WfT_glove[Xcols_comb],WfT_glove[Ycols]
    
    '''combine the Wf from titles with sentiment and topics from body'''
    WfT_senti_topic = pd.concat([WfT_X,Senti,Topic,WfT_Y],axis=1)
    WfT_senti_topic = WfT_senti_topic.dropna()
    Xcols_comb = XcolsT+list(Senti.columns)+list(Topic.columns)
    CombineFeatDict['WfT_senti_topic_X'],CombineFeatDict['WfT_senti_topic_Y'] = WfT_senti_topic[Xcols_comb],WfT_senti_topic[Ycols]
    
    '''combine the Wf from titles with sentiment and glove from body'''
    WfT_senti_glove = pd.concat([WfT_X,Senti,Glove,WfT_Y],axis=1)
    WfT_senti_glove = WfT_senti_glove.dropna()
    Xcols_comb = XcolsT+list(Senti.columns)+list(Glove.columns)
    CombineFeatDict['WfT_senti_glove_X'],CombineFeatDict['WfT_senti_glove_Y'] = WfT_senti_glove[Xcols_comb],WfT_senti_glove[Ycols]
    
    '''combine the Wf from titles with topic and glove from body'''
    WfT_topic_glove = pd.concat([WfT_X,Topic,Glove,WfT_Y],axis=1)
    WfT_topic_glove = WfT_topic_glove.dropna()
    Xcols_comb = XcolsT+list(Topic.columns)+list(Glove.columns)
    CombineFeatDict['WfT_topic_glove_X'],CombineFeatDict['WfT_topic_glove_Y'] = WfT_topic_glove[Xcols_comb],WfT_topic_glove[Ycols]
    
    '''combine the Wf from titles with sentiment,topic, glove from body'''
    WfT_senti_topic_glove = pd.concat([WfT_X,Senti,Topic,Glove,WfT_Y],axis=1)
    WfT_senti_topic_glove = WfT_senti_topic_glove.dropna()
    Xcols_comb = XcolsT+list(Senti.columns)+list(Glove.columns)+list(Topic.columns)
    CombineFeatDict['WfT_senti_topic_glove_X'],CombineFeatDict['WfT_senti_topic_glove_Y'] = WfT_senti_topic_glove[Xcols_comb],WfT_senti_topic_glove[Ycols]
    
    ComNameList = ['WfTDB','WfT_Senti','WfT_Topic','WfT_Glove',
                   'WfT_senti_topic','WfT_senti_glove','WfT_topic_glove','WfT_senti_topic_glove']
    
    return CombineFeatDict, ComNameList
           
def getCombineFeat_NIE(Week_tem_NLP_dict,selectedFeat):
    CombineFeatDict = dict()
    '''WF: word frequency, SE: sentiment, TD:topic distribution, GW: Glove Word embedding'''
    '''combine the word frequency of titles, des and body together'''    
    Load_X = Week_tem_NLP_dict['YXtable_week_temp_X']
    Load_Y = Week_tem_NLP_dict['YXtable_week_temp_Y']
    
    Ycols = list(Load_Y.columns)
    Xcols = list(Load_X.columns)
    
    Count = Week_tem_NLP_dict['body_features_count_X'][selectedFeat['body_features_count']]
    Senti = Week_tem_NLP_dict['body_features_senti_X'][selectedFeat['body_features_senti']]
    # WordFreq = Week_tem_NLP_dict['title_WordFreqDf_X'][selectedFeat['title_WordFreqDf']]
    Topic = Week_tem_NLP_dict['body_topicDistributionDf_grouped_X'][selectedFeat['body_topicDistributionDf_grouped']]
    # Glove = Week_tem_NLP_dict['title_features_Glove_grouped_X'][selectedFeat['title_features_Glove_grouped']]
    
    # according to the best-performing to select
    
    '''combine the Count-B, senti-B, Topic-B'''  
    Count_Senti_Topic = pd.concat([Load_X,Count,Senti,Topic,Load_Y],axis=1)
    Count_Senti_Topic = Count_Senti_Topic.dropna()
    Xcols_comb = Xcols+list(Count.columns)+list(Senti.columns)+list(Topic.columns)
    CombineFeatDict['Count_Senti_Topic_X'],CombineFeatDict['Count_Senti_Topic_Y'] = Count_Senti_Topic[Xcols_comb],Count_Senti_Topic[Ycols]
        
    ComNameList = ['Count_Senti_Topic']
    
    return CombineFeatDict, ComNameList

def trainModels_for_add_tempeature_with_NLP(configuration,lags,Week_temp_NLP_dict,selectedFeat,ComNameList):
    
    try:   
        # tableList = ['YXtable_week_temp'] + configuration['NLP_features_table']
        tableList = ComNameList
        ResultsDict = dict()
        
        for table in tableList:
            print(table)
            tableDict = dict()
            # splitting data
            X,Y = Week_temp_NLP_dict[table+'_X'],Week_temp_NLP_dict[table+'_Y']
            Xcols,Ycols = X.columns,Y.columns
            Xidx, Yidx = X.index,Y.index
            X = pd.DataFrame(np.nan_to_num(X),columns=Xcols)
            Y = pd.DataFrame(np.nan_to_num(Y),columns=Ycols)
            X.index,Y.index = Xidx,Yidx
            
            X_train,X_valid,X_test,Y_train,Y_valid,Y_test,scaler_x,scaler_y= getTrainTest(X,Y,configuration)
            
            X_train_S2 = pd.concat([X_train,X_valid])
            Y_train_S2 = pd.concat([Y_train,Y_valid])
            
            #training model and get the feature importance
            ETree_S2 = tuningCV_ExtraTrees(X_train_S2, Y_train_S2, configuration)
            
            del X,Y,X_train,Y_train,Y_valid,X_train_S2, Y_train_S2
            
            
            dfFeatImp = pd.DataFrame({'name': ETree_S2.feature_names_in_, 
                                      'value': ETree_S2.feature_importances_}).sort_values(by=['value'])
            
            
            
            #forecasting
            for_truth = ExtraTreePre(ETree_S2,X_test,Y_test,scaler_y)
            ForecastError = calculate_error(for_truth[['Truth','Forecasts']],'Truth','Forecasts')
            
            del X_test,Y_test,X_valid,ETree_S2,scaler_y
            tableDict['Feature_importance'] = dfFeatImp
            tableDict['Forecasts'] = for_truth
            del dfFeatImp
            tableDict['Errors'] = ForecastError

            ResultsDict[table] = tableDict
            del tableDict
        
        return ResultsDict
    
    except Exception as e:
        logError(e)

## Evaluation
def getErrorMean_benchmark(emdName,Final_dict):
    rmseL,maeL,smapeL = [],[],[]
    for i in range(10):
        error = Final_dict['Times'+str(i)][emdName]
        rmseL.append(error[0])
        maeL.append(error[1])
        smapeL.append(error[2])
    print(emdName)
    print('average RMSE is', np.mean(rmseL))
    print('std RMSE is', np.std(rmseL))
    print('average MAE is', np.mean(maeL))
    print('std MAE is', np.std(maeL))
    print('average SMAPE is', np.mean(smapeL))
    print('std SMAPE is', np.std(smapeL))

# get the error mean and std for all the ten times
def getErrorMean(emdName,Final_dict):
    rmseL,maeL,smapeL = [],[],[]
    for i in range(10):
        error = Final_dict['Times'+str(i)][emdName]['Errors']
        rmseL.append(error.rmse.mean())
        maeL.append(error.mae.mean())
        smapeL.append(error.smape.mean())
    print(emdName)
    print('average RMSE is', np.mean(rmseL))
    print('std RMSE is', np.std(rmseL))
    print('average MAE is', np.mean(maeL))
    print('std MAE is', np.std(maeL))
    print('average SMAPE is', np.mean(smapeL))
    print('std SMAPE is', np.std(smapeL))

def getErrorMean_section(emdName,Final_dict,section):
    rmseL,maeL,smapeL = [],[],[]
    for i in range(10):
        error = Final_dict['Times'+str(i)][emdName]['Errors']
        rmseL.append(error.rmse.mean())
        maeL.append(error.mae.mean())
        smapeL.append(error.smape.mean())
    content = f"{emdName}\n"
    content += f"average RMSE is {np.mean(rmseL)}\n"
    content += f"std RMSE is {np.std(rmseL)}\n"
    content += f"average MAE is {np.mean(maeL)}\n"
    content += f"std MAE is {np.std(maeL)}\n"
    content += f"average SMAPE is {np.mean(smapeL)}\n"
    content += f"std SMAPE is {np.std(smapeL)}\n"
    return content
    

#%%            
warnings.filterwarnings('ignore')
configuration = auxConfig.load_forecasting_NLP()

#Training model selecting on test set considering holidays and temperature
YXtableDict,FeatDict = createXYtables4training(configuration,48,NIE=False)

# Train benchmark models
Bench_dict = dict()
for i in range(10):
    print('*'*50)
    print('Times-'+str(i))
    print('*'*50)
    selectedResults = trainModels_for_selecting_on_test(configuration,lags=48,YXtableDict,FeatDict)
    Bench_dict['Times'+str(i)] = selectedResults

# Evaluate results
NameList = ['benchmark','YXtable_NoText','YXtable_holiday', 'YXtable_weekend',
            'YXtable_temperature','YXtable_week_temp','YXtable_week_hol_temp']
for name in NameList:
    getErrorMean_benchmark(name,Bench_dict)

#%%
# Get textual features selected by Granger test
Week_tem_NLP_dict,selectedFeat = getNLP_for_week_tem(YXtableDict,configuration,48,electric=False,section=None)

# Get AIC values for granger lags
GrangerLagDf = GrangerLagsSelect(YXtableDict,configuration,48,selectedFeat)

#%%
# Forecasting with combined textual features -- for 10 times
CombineFeatDict, ComNameList = getCombineFeat(Week_tem_NLP_dict,selectedFeat)
CombResults = trainModels_for_add_tempeature_with_NLP(configuration,48,CombineFeatDict,selectedFeat,ComNameList)
    
#%%
# LIME local explaination
def extract_word_in_keys(coff_dict):
    """
    Extract words from keys in a dictionary and reassign values to new keys
    """
    key_list = list(coff_dict.keys())
    for key in key_list:
        words = key.split(' ')
        for word in words:
            if any(c.isalpha() for c in word):
                coff_dict[word] = coff_dict.pop(key)
    return coff_dict

def prepare_data(combine_feat_dict, configuration):
    """
    Prepare training and test datasets
    """
    X1 = combine_feat_dict['WfTDB_X'][['driver-T','mps-D','power-B','coronavirus-T','pandemic-D','coronavirus-D']]
    X2 = combine_feat_dict['WfT_senti_topic_glove_X'][['subject_body_min','Topic-5','Topic-18','dim9','dim51','dim69']]
    X3 = combine_feat_dict['WfT_senti_topic_glove_X'].iloc[:,:58]
    X = pd.concat([X3, X1, X2], axis=1)
    Y = combine_feat_dict['WfT_senti_topic_glove_Y']
    
    Xcols, Ycols = X.columns, Y.columns
    Xidx, Yidx = X.index, Y.index
    X = pd.DataFrame(np.nan_to_num(X), columns=Xcols)
    Y = pd.DataFrame(np.nan_to_num(Y), columns=Ycols)
    X.index, Y.index = Xidx, Yidx
    
    return get_train_test(X, Y, configuration)

def train_and_predict(X_train, X_valid, X_test, Y_train, Y_valid, configuration):
    """
    Train the model and make predictions
    """
    X_train_S2 = pd.concat([X_train, X_valid])
    Y_train_S2 = pd.concat([Y_train, Y_valid])
    
    model = tuningCV_ExtraTrees(X_train_S2, Y_train_S2, configuration)
    Y_pred = model.predict(X_test)
    
    return model, Y_pred

def explain_predictions(model, X_train_S2, X_test, subkey):
    """
    Use LIME to explain model predictions
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_S2.values, feature_names=X_train_S2.columns.values.tolist(),
        class_names=['Forecasts'], verbose=True, mode='regression'
    )
    
    lime_df = pd.DataFrame(columns=subkey)
    for i in range(len(X_test)):
        print(X_test.index[i])
        exp = explainer.explain_instance(X_test.values[i], model.predict, num_features=len(X_test.columns))
        coff_dict = dict(exp.as_list())
        coff_dict = extract_word_in_keys(coff_dict)
        selected_feat = {key: coff_dict.get(key, 0) for key in subkey}  # Handle missing keys safely
        lime_df = lime_df.append(selected_feat, ignore_index=True)
        print(lime_df.shape)
    
    lime_df.index = X_test.index
    return lime_df

X_train, X_valid, X_test, Y_train, Y_valid, Y_test, _, _ = prepare_data(combine_feat_dict, configuration)
model, Y_pred = train_and_predict(X_train, X_valid, X_test, Y_train, Y_valid, configuration)

subkey = ['driver-T','mps-D','power-B','coronavirus-T','pandemic-D','coronavirus-D',
          'subject_body_min','Topic-5','Topic-18','dim9','dim51','dim69']
lime_df = explain_predictions(model, X_train.append(X_valid), X_test, subkey)

#%%
'''experiments on test word2vec, tfidf, GloVe'''
Embedding_NLP_dict,selEmbedding = getEmbeddings_for_week_tem(Week_tem_NLP_dict,configuration,48)
# Embedding_NLP_dict is the tables with tfidf and word2vec features
# selEmbedding is the selected tfidf and word2vec features

# train the model
emdNameList = ['groupedTfidf.csv','groupedW2Vdf.csv']
Final_dict_embedding = dict()
for i in range(10):
    print('*'*50)
    print('Times-'+str(i))
    print('*'*50)
    embeddingResults = trainModels_for_add_tempeature_with_NLP(configuration,48,Embedding_NLP_dict,selEmbedding,emdNameList)
    Final_dict_embedding['Times'+str(i)] = embeddingResults  

# Evaluate
for emdName in emdNameList:
    getErrorMean(emdName,Final_dict_embedding)

#%%
'''experiments on electricity-related news'''
Week_tem_NLP_dict_electric,selectedFeat = getNLP_for_week_tem(Week_tem_NLP_dict,configuration,48,electric=True)

def changeDictName(oldDict,electric=False):
    if electric==False:
        newKey = ['count','wordFreq','sentiment','topic','glove']
    else:
        newKey = ['YXtable_week_temp_X','YXtable_week_temp_Y',
                  'count_Y','count_X','wordFreq_Y','wordFreq_X','sentiment_Y','sentiment_X',
                  'topic_Y','topic_X','glove_Y','glove_X']
    oldKey = list(oldDict.keys())
    newDict = dict()
    for i in range(len(oldDict.keys())):
        newDict[newKey[i]] = oldDict[oldKey[i]]
    return newDict

selectedFeat_rename = changeDictName(selectedFeat,electric=False)
Week_tem_NLP_dict_electric_rename = changeDictName(Week_tem_NLP_dict_electric,electric=True)

# add the glove and wordfreq combination to the table dict
electricGloveFeat = Week_tem_NLP_dict_electric_rename['glove_X'][selectedFeat_rename['glove']]
combGloveWord = Week_tem_NLP_dict_electric_rename['wordFreq_X'].merge(electricGloveFeat,left_on='Date', right_on='Date', how='left')
Week_tem_NLP_dict_electric_rename['comb_X'] = combGloveWord
Week_tem_NLP_dict_electric_rename['comb_Y'] = Week_tem_NLP_dict_electric_rename['glove_Y']
    
# train the model
electricNameList = ['wordFreq','glove','comb']
Final_dict_electric = dict()
for i in range(10):
    print('*'*50)
    print('Times-'+str(i))
    print('*'*50)
    electricResults = trainModels_for_add_tempeature_with_NLP(configuration,48,Week_tem_NLP_dict_electric_rename,selectedFeat_rename,electricNameList)
    Final_dict_electric['Times'+str(i)] = electricResults

for name in electricNameList:
    getErrorMean(name,Final_dict_electric)
    

#%%
'''Experiments on the North Ireland data with all the news and temperatures from Belfast'''
# generate data
NIE_YXtableDict,NIE_FeatDict = createXYtables4training(configuration,lags=48,NIE=True)

# train benchmark models
Bench_dict_NIE = dict()
for i in range(10):
    print('*'*50)
    print('Times-'+str(i))
    print('*'*50)
    selectedResults = trainModels_for_selecting_on_test(configuration,48,NIE_YXtableDict,NIE_FeatDict)
    Bench_dict_NIE['Times'+str(i)] = selectedResults

NIENameList = ['benchmark','YXtable_NoText','YXtable_holiday', 'YXtable_weekend',
               'YXtable_temperature','YXtable_week_temp','YXtable_week_hol_temp']
for name in NIENameList:
    getErrorMean_benchmark(name,Bench_dict_NIE)
    
# add textual features
Week_tem_NLP_dict_NIE,selectedFeat_NIE = getNLP_for_week_tem(NIE_YXtableDict,configuration,48,electric=False)
    
# train with separated textual features
NIEFeatNameList = ['title_features_count','title_WordFreqDf','title_topicDistributionDf_grouped',
                   'title_features_Glove_grouped',
                   'des_features_count','des_WordFreqDf','des_topicDistributionDf_grouped',
                   'des_features_Glove_grouped',
                   'body_features_count','body_WordFreqDf',
                   'body_features_senti','body_topicDistributionDf_grouped']
SeparateNIE_text_dict = dict()
for i in range(10):
    print('*'*50)
    print('Times-'+str(i))
    print('*'*50)
    sepNIEResults = trainModels_for_add_tempeature_with_NLP(configuration,48,Week_tem_NLP_dict_NIE,selectedFeat_NIE,NIEFeatNameList)
    SeparateNIE_text_dict['Times'+str(i)] = sepNIEResults

for name in NIEFeatNameList:
    getErrorMean(name,SeparateNIE_text_dict)

#%% 
'''Experiments on the other section news'''
sectionList = ['UK', 'Business', 'UK Politics', 'Europe', 'Entertainment & Arts',
               'US & Canada', 'Wales', 'London', 'Health', 'Northern Ireland','Electric']

for section in sectionList:  
    print(section)
    # select textual features
    Week_tem_NLP_dict_section,selectedFeat = getNLP_for_week_tem(Week_tem_NLP_dict,configuration,48,electric=False,section=section)
    
    # get the textual features groups that not empty
    NameList,keys = [],list(selectedFeat.keys())
    for k in keys:
        if selectedFeat[k] != []:
            NameList.append(k)
    
    # train the model with textual features
    if NameList != []:
        Final_dict = dict()
        for i in range(10):
            print('*'*50)
            print('Times-'+str(i))
            print('*'*50)
            sectionResults = trainModels_for_add_tempeature_with_NLP(configuration,48,Week_tem_NLP_dict_section,selectedFeat,NameList)
            Final_dict['Times'+str(i)] = sectionResults
    




