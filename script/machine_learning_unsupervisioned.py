"""
AUTHOR: EMERSON CAMPOS BARBOSA JÚNIOR
UNSUPERVISIONED MACHINE LEARNING
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

### reading and verifying data ###

data_general = pd.read_csv('data/Mall_Customers.csv')

data_general.info()
data_general.isna().sum()
data_general.shape
data_general.columns

encoder = LabelEncoder()

data_general['Gender'] = encoder.fit_transform(data_general['Gender'])

X = data_general.drop(['CustomerID', 'Gender', 'Age'], axis=1)
X = X.values

### model 1 ###
model1 = KMeans(n_clusters=5, init='k-means++', random_state=1)
y_kmeans = model1.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

### model 2 ###
pca = PCA()
components = pca.fit_transform(X)
plt.scatter(components[:, 0], components[:, 1], c=data_general['Age'])
