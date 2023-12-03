import math 
import random
import pandas as pd 
import numpy as np  
import pickle 
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor 
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler  
import database as db 


def calculate_lifetime(pickup_date, birthday): 
    return round((pickup_date - birthday).total_seconds()/3600/24, 3) 


def merge_prediction_prob(pred, prob): 
    return "{} ({:.2f}%)".format(pred, prob*100)


def predict_ragout_group(df): 
    df["usage_period"]          = df.apply(lambda x: calculate_lifetime(x['last_updated_date'], x['birthday']), axis=1)
    df['usage_period_laundris'] = df.apply(lambda x: calculate_lifetime(x['last_updated_date'], x['creation_date']), axis=1) 

    features = ['item_type_id', 
                'customer_id', 
                'total_washes', 
                'pickup_count', 
                'dropoff_count', 
                'usage_period', 
                'usage_period_laundris']  
    
    scaler_filename = "models/ragout_classification_scaler.pkl" 
    model_filename = "models/ragout_classification_model_v1.pkl"  

    scaler = MinMaxScaler() 
    scaler = pickle.load(open(scaler_filename, 'rb')) 
    model_lgbm = pickle.load(open(model_filename, 'rb')) 

    data = scaler.transform(df[features]) 

    df['prediction'] =  model_lgbm.predict(data) 
    probabilities = model_lgbm.predict_proba(data) 
    df['prediction_confidence'] = np.max(probabilities, axis=1)

    df["predicted_ragout"]= df.apply(lambda x: merge_prediction_prob(x['prediction'], x['prediction_confidence']), axis=1) 

    return df[['rfid_id', 'usage_period', 'prediction','predicted_ragout']]
