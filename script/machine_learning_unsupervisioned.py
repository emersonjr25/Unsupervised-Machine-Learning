"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
UNSUPERVISIONED MACHINE LEARNING
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import LabelEncoder

data_general = pd.read_csv('data/Assignment-1_Data.csv', sep=';')

#country maybe after
data_general = data_general.drop(['BillNo', 'Itemname', 'Date'], axis = 1)

data_general.isna().sum()

data_general = data_general.dropna()

encoder = LabelEncoder()
data_general['Country'] = encoder.fit_transform(data_general['Country'])
### exploring data ###

list = []

for i in range(data_general.shape[1]):
    list.append(len(data_general.iloc[:, i].unique()))
    print(list)

data_general.info()
data_general.describe()

data_general = data_general[data_general['Quantity'] <= 5000]

array = np.array(data_general['Price'])
list_transform = [i.replace(',', '.') for i in array]

data_general['Price'] = np.array(list_transform).astype(float)

data_general.info()
### organizing data ###

### model 1 ###
pca = PCA(n_components=2)
components = pca.fit_transform(data_general)
plt.scatter(components[:, 0], components[:, 1])

### model 2 ###
k_means = KMeans(n_clusters=2)
k = k_means.fit_transform(data_general)
plt.scatter(k[:, 0], k[:, 1], c=data_general['Country'])
### model 3 ###
dbscan = DBSCAN()
db = dbscan.fit_predict(data_general)
plt.hist(db)
