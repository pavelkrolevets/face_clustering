# Face clustering using FaceNet and KMeans

### Prerequisites

* correct ```PYTHONPATH=full_path_to_folder/scr```
* pretrained FaceNet model
in the folder ```./models/```. You can download it [here](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk).
* images dataset in the ```./data/folder_with_images/```. I used [LFW dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz). Unzip it to ```./data/lfw/```.


### How to run
1.  Clean folders ```./np_embeddings``` ,  ```./data/clustered```, ```./data/sorted```

* Run ```export_embeddings.py``` </br>
This will generate embeddings and labels from images.</br>
To make it working add ```PYTHONPATH=full_path_to_folder/scr``` to you sources. If you are using PyCharm simply add to Enviroment Varianbles ```PYTHONPATH``` value ```full_path_to_folder/scr```.

* Run ```Distance_matrix.py```.
</br> This step will give you the matrix with all of the Eculidian distances between faces. Numpy array is saved to  ```./np_embeddings/embeddings.npy```
It will take some time because the manrix size of ```NxN``` with zeros on the main diagonal. This file is quiet big - 1.4GB. We need it to sort faces.

3. Run ```Cluster_faces.py```.
</br> First, it will sort all off the face images based on the closest distance and save sorted images to ./data/sorted.
Second, it will cluster images using Kmeans algorithm. Number of cluster by default is 30. You can change it if you like.

## Disclaimer
This work is based on FaceNet achievement. You can check FaceNet model and papers [here](https://github.com/davidsandberg/facenet).
