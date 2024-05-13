# Preprocessing

import numpy as np
import pandas as pd

# Unnamed: 0                  int64
#  Date Published             object
#  Overall Rating            float64
#  Passenger Country          object
#  Trip_verified              object
#  Comment title              object
#  Comment                    object
#  Aircraft                   object
#  Type Of Traveller          object
#  Seat Type                  object
#  Origin                     object
#  Destination                object
#  Date Flown                 object
#  Seat Comfort              float64
#  Cabin Staff Service       float64
#  Food & Beverages          float64
#  Ground Service            float64
#  Value For Money           float64
#  Recommended                object
#  Inflight Entertainment    float64
#  Wifi & Connectivity       float64


# Import data
def import_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['Unnamed: 0'])
    return data

# Encode categorical variables
def encode_categoricals(data):
    for column in ['Passenger Country', 'Trip_verified', 'Aircraft', 'Type Of Traveller', 'Seat Type', 'Origin', 'Destination', 'Recommended']:
        # get dummies
        data[column] = pd.get_dummies(data[column])
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


