"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
UNSUPERVISIONED MACHINE LEARNING
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

data_general = pd.read_csv('data/Assignment-1_Data.csv', sep=';')

data_general.isna().sum()

data_general = data_general.dropna()
data_general = data_general.drop('Date', axis = 1)

### exploring data ###

list = []

for i in range(data_general.shape[1]):
    list.append(len(data_general.iloc[:, i].unique()))
    print(list)

data_general['Country'].unique()
data_general.info()
data_general.describe()
data_general['BillNo'] = data_general['BillNo'].astype(float)

### model 1 ###
pca = PCA(n_components=2)
pca.fit_transform(data_general)

### model 2 ###
k_means = KMeans(n_clusters=7)
k_means.fit_transform(data_general)
