import pandas as pd
import numpy as np
import opendatasets as opd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path = 'data/heart-failure-prediction/heart.csv'

if not os.path.exists(path):
    url = 'https://www.kaggle.com/fedesoriano/heart-failure-prediction'
    op = opd.download(url,'data')

def dataframe_to_arrays(dataframe):
    # chuyển dataframe về dạng số 
    dataframe1 = dataframe.copy(deep=True)
    categorical_cols = [col_name for col_name in dataframe.select_dtypes(exclude='number')]
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    inputs = dataframe1[[col for col in dataframe1.columns[:-1]]].to_numpy()
    targets = dataframe1[dataframe.columns[-1]].to_numpy()
    return inputs, targets

if __name__ == '__main__':
    dataframe_raw = pd.read_csv(path)
    X, y = dataframe_to_arrays(dataframe_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    clf = RandomForestClassifier(max_depth=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))