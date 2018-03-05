import numpy as np

embeddings = np.load('./np_embeddings/embeddings.npy') # load embeddings of faces
nrof_images = len(embeddings)
matrix = np.zeros((nrof_images, nrof_images))

print('')
# Print distance matrix
print('Distance matrix')
print('    ', end='')
for i in range(nrof_images):
    print('    %1d     ' % i, end='')
print('')
for i in range(nrof_images):
    #print('%1d  ' % i, end='')
    for j in range(nrof_images):
        dist = np.sqrt(np.sum(np.square(np.subtract(embeddings[i, :], embeddings[j, :]))))
        matrix[i][j] = dist
    if i % 1000 == 0:
        print('  %1.4f  ' % i, '\n', end='')

print('')
np.save('./np_embeddings/matrix.npy', matrix)
print("Saved")


