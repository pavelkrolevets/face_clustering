# Face clustering using FaceNet and KMeans

Before startinng clean folders ./np_embeddings ,  ./data/clustered, ./data/sorted

1. Run export_embeddings.py to generate embeddings and labels for images.
To make it working add PYTHONPATH=full_path_to_this_folder/scr to you sources. If you are using PyCharm simply add in Enviroment Varianbles PYTHONPATH value 'full_path_to_this_folder/scr'
If you dont have a good graphics card installed on the machine it can take some time :-)

2. Run Distance_matrix.py. This step will give you the matrix with all of the faces distances. File is saved to  './np_embeddings/embeddings.npy'
It will take some time because the amount of computations is O(x^2). This file is quiet big - 1.4GB. We need it to sort faces.

3. Finally, Run Cluster_faces.py. First, it will sort all off the face images based on the closest distance and save it to ./data/sorted.
Second, it will cluster images using Kmeans algorithm. Number of cluster by default is 30. You can change it if youd like.


