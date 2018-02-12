#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 13:19:26 2017

@author: luogan
"""

import pandas as pd
df = pd.read_csv('loans.csv')

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
dff =df.apply(lambda df: d[df.name].fit_transform(df))
dff.to_excel('dff.xls')



import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
 
import matplotlib.pylab as plt
#%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
 
train = pd.read_excel('dff.xls')
target = 'safe_loans'
IDcol = 'id'


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['safe_loans'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    from pandas import DataFrame
    '''
    gg=DataFrame(dtrain_predictions)
    gg.to_excel('dtrain_predictions.xls')   
    
    tt=DataFrame(dtrain_predprob)
    tt.to_excel('dtrain_predprob.xls')
    '''
    
    print(alg)
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['safe_loans'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['safe_loans'], dtrain_predprob))
    
    ww=(alg.feature_importances_)
    print(ww)            
    feat_imp = pd.Series(ww).sort_values(ascending=False)
    
    #print(feat_imp)
    
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    
    """
    model=alg
    featureImportance = model.get_score() 
    features = pd.DataFrame() 
    features['features'] = featureImportance.keys() 
    features['importance'] = featureImportance.values() 
    features.sort_values(by=['importance'],ascending=False,inplace=True) 
    fig,ax= plt.subplots() 
    fig.set_size_inches(20,10) 
    plt.xticks(rotation=60) 
    #sn.barplot(data=features.head(30),x="features",y="importance",ax=ax,orient="v") 
    """
    
    
#Choose all predictors except target & IDcols
predictors = [x for x in train.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
                learning_rate =0.1,
                n_estimators=1000,
                max_depth=18,
                min_child_weight=1,
                gamma=0,
                subsample=0.8,
                colsample_bytree=0.8,
                objective= 'binary:logistic',
                nthread=4,
                scale_pos_weight=1,
                seed=27)
modelfit(xgb1, train, predictors)     




 