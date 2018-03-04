import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import pandas as pd
from shutil import copy2
import os

names = list(np.load('./np_embeddings/image_list.npy'))
embeddings = pd.DataFrame(np.load('./np_embeddings/embeddings.npy'), index=names)
matrix_ndarr = np.load('./np_embeddings/matrix.npy')

matrix = pd.DataFrame(data=matrix_ndarr, index=names, columns=names)
# """Example to find the closest faces
# In case of recomendation we simply can get a face with the lowest Euclidian distance to a seen face
# and show to a user"""
sorted_faces = []
face = names[0]
sorted_faces.append(face)
for i in range(len(embeddings)-1):
    closest_faces = matrix.loc[[face]].squeeze() # row number, you can find who is it in the row of names
    closest_faces=closest_faces.drop(index=sorted_faces,  inplace=False, errors='raise')
    closest_faces = closest_faces.sort_values()
    similar_face = closest_faces.index[0]
    sorted_faces.append(similar_face)
    face = similar_face
    print(face)
for i in range(len(sorted_faces)):
    folder_for_sort = './data/sorted/'
    folder_for_sort = folder_for_sort + str([i])
    file = sorted_faces[i]
    print(file)
    copy2(file, folder_for_sort)
"""Clustering"""
# K means works quiet well in this case.
# Notice the more clusters - the closer faces in each cluster
kmeans = KMeans(n_clusters=30, random_state=40, max_iter=100000)
kmeans.fit(embeddings)
labels_kmeans = kmeans.labels_
"""Saving images to cluster folders"""
for i in range(len(labels_kmeans)):
    folder_for_clust = './data/clustered/'
    folder_for_clust = folder_for_clust + str(labels_kmeans[i])+'/'
    if not os.path.exists(folder_for_clust):
        os.makedirs(folder_for_clust)
    file = names[i]
    print(file)
    copy2(file, folder_for_clust)
print("Done!")


