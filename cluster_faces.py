"""
Author: Pavel Krolevets @ Shanghai Jiao Tong University
E-mail: pavelkrolevets@gmail.com

Step 3 of the pipeline: turn the distance matrix into a browsable ordering
and clusters of similar faces.

Theory
------
1. Nearest-neighbor chain (greedy "sort by likeness").
   Start from the first image; repeatedly pick the *not-yet-sorted* image
   with the smallest distance to the current one. Consecutive images in the
   resulting chain are maximally similar, so photos of the same person
   appear as contiguous runs. The chain depends on the starting image (the
   ordering is not unique) but is deterministic once the matrix is fixed.

2. KMeans clustering.
   KMeans partitions the 128-d embedding space into `n_clusters` regions:
   assign every point to its nearest centroid, move each centroid to the
   mean of its assigned points, and repeat until it stops changing.
   FaceNet embeddings vary mostly along the *identity* axis (the network
   was trained to pull same-person images together), so the cells
   approximate person groups. There is no guarantee that one cluster =
   one person: a person's images can split across neighbouring cells
   (pose, lighting, expression) and a cell can mix a few look-alike
   people. More clusters => smaller cells => tighter groups, at the cost
   of more within-person splits.

Outputs:
  ./data/sorted/     images in chain order (files 0..N-1, byte-identical copies)
  ./data/clustered/  one folder per KMeans cluster, original filenames kept
"""

import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from shutil import copy2
import os

# Inputs from the previous steps. `names` is the source path of every
# image; the embedding rows and the matrix rows/columns share exactly this
# order, so the three arrays can always be combined row-by-row.
names = list(np.load('./np_embeddings/image_list.npy'))
embeddings = pd.DataFrame(np.load('./np_embeddings/embeddings.npy'), index=names)
matrix_ndarr = np.load('./np_embeddings/matrix.npy')

# Label the distance matrix with the paths so rows can be looked up by image.
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
# Save the chain: image i of the chain becomes file ./data/sorted/<i>.
for i in range(len(sorted_faces)):
    folder_for_sort = './data/sorted/'
    folder_for_sort = folder_for_sort + str(i)
    file = sorted_faces[i]
    print(file)
    copy2(file, folder_for_sort)
"""Clustering"""
# K means works quiet well in this case.
# Notice the more clusters - the closer faces in each cluster
# random_state fixes the initial centroid placement, so the clusters are
# identical from run to run on the same embeddings.
kmeans = KMeans(n_clusters=30, random_state=40, max_iter=100000)
kmeans.fit(embeddings)
labels_kmeans = kmeans.labels_  # cluster id of every image, in `names` order
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


