# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cluster_tools as ct
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn import metrics as skmet

df_co2 = pd.read_csv("csvca2.csv")
print(df_co2.describe())
df_co3 = df_co2[["1960","1970", "1980", "1990", "2000", "2010", "2017"]]
print(df_co3.describe())
df_co3 = df_co3.drop(["1960"], axis=1)
print(df_co3.describe())


corr = df_co3.corr()
print(corr)


ct.map_corr(df_co3)
plt.show()


pd.plotting.scatter_matrix(df_co3, figsize=(12, 12), s=5, alpha=0.8)
plt.show()
df_ex = df_co3[["2000", "2017"]] # extract the two columns for clustering
df_ex = df_ex.dropna() # entries with one nan are useless
df_ex = df_ex.reset_index()
print(df_ex.iloc[0:15])


df_norm, df_min, df_max = ct.scaler(df_ex)
print()
print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
# set up the clusterer with the number of expected clusters
  kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
  kmeans.fit(df_norm) # fit done on x,y pairs
  labels = kmeans.labels_
# extract the estimated cluster centres
  cen = kmeans.cluster_centers_
# calculate the silhoutte score
  print(ncluster, skmet.silhouette_score(df_ex, labels))





# reset_index() moved the old index into column index
# remove before clustering
#df_ex = df_ex.drop("index", axis=1)

# reset_index() moved the old index into column index
# remove before clustering

ncluster = 9

 # best number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)


#scaler = StandardScaler()
#df_norm = scaler.fit_transform(df_ex)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

plt.scatter(df_norm["2000"], df_norm["2017"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("CO2(2000)")
plt.ylabel("CO2(2017)")
plt.show()

print(cen)
df_cluster = df_norm[["2000", "2017"]].copy()
# Applying the backscale function to convert the cluster centre
df_cluster, df_min, df_max = ct.scaler(df_cluster)

scen = ct.backscale(cen, df_min, df_max)

print(scen)

# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')

plt.scatter(df_ex["2000"], df_ex["2017"], 10, labels, marker="o", cmap=cm)
plt.scatter(xcen, ycen, 45, "k", marker="d")
plt.xlabel("CO2(2000)")
plt.ylabel("CO2(2017)")
plt.show()
