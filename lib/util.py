"""
    @name: util.py
    @description: A module of utility functions used in Jupyter Notebooks
"""

import pandas as pd
import numpy as np

def fetch_tweets(event):
    """ Read a CSV file with cleaned PHEME event tweets
    
    Note: 
        - Setting engine to "python" helps with large datasets
    
    Params:
        - event {str} the name of the event
    
    Return: a Pandas dataframe
    
    """
    return pd.read_csv("data/tweets/%s.csv" % event, 
                 dtype={
                    'tweet_id': str,
                    'in_reply_tweet': str,
                    'thread': str,
                    'user_id': str,
                    'in_reply_user': str
                 },
                 engine="python")


def fetch_extracted(event, is_normalized=True):
    """ Return dataset X and results vector y 
    
    Params:
        - event {str} the name of the event in the PHEME dataset
        - is_normalized {bool} returned X matrix as normalized. Deafult is True
    """ 
    X = pd.read_csv("data/analyzed/converted/%s.csv" % event, engine="python")
    y = X.rumorspreader
    X = X.drop(["rumorspreader"], axis=1)
    if is_normalized:
        X = (X - X.mean()) / X.std()
    return X, y
