# Preprocessing

import numpy as np
import pandas as pd

# Import data
def import_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed: 0'])
    data['Trip_verified'] = data['Trip_verified'].replace({'NotVerified': 'Not Verified', 'Unverified': 'Not Verified'})
    return data

# Create datetime object & variables
def create_datetime(data):
    data['Date Published'] = pd.to_datetime(data['Date Published'])
    data['Date Flown'] = pd.to_datetime(data['Date Flown'])
    
    data['Year Published'] = data['Date Published'].dt.year
    data['Month Published'] = data['Date Published'].dt.month
    data['Day Published'] = data['Date Published'].dt.day
    
    data['Year Flown'] = data['Date Flown'].dt.year
    data['Month Flown'] = data['Date Flown'].dt.month
    data['Day Flown'] = data['Date Flown'].dt.day
    return data

# Encode categorical variables
def encode_categoricals(data):
    # encode all columns except for Overall rating, Comment title, Comment
    for col in data.columns:
        if col not in ['Comment title', 'Comment', 'Overall Rating', 'Date Published', 'Date Flown']:
            data = pd.get_dummies(data, columns=[col], prefix=col)
    return data

# Drop highly correlated column (see exploration.py) -> nothing to be dropped b/c all correlations < abs(0.8)

# Drop missing values in target 'Overall Rating'
def drop_missing_target(data):
    data = data.dropna(subset=['Overall Rating'])
    return data

# Normalization of data not needed because there are no continuous features

# Create pipeline with all preprocessing steps
def create_pipeline(file_path):
    data = import_data(file_path)
    data = create_datetime(data)
    data = encode_categoricals(data)
    data = drop_missing_target(data)
    return data

# To be added once finished:
# - new features from sentiment analysis
# - new features from topic modeling





