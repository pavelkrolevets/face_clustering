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
    # Vectorized per row: computes exactly the same Euclidean distances as
    # the original element-wise double loop, but as a single numpy operation.
    matrix[i, :] = np.sqrt(np.sum(np.square(embeddings[i, :] - embeddings), axis=1))
    if i % 1000 == 0:
        print('  %1.4f  ' % i, '\n', end='')

print('')
np.save('./np_embeddings/matrix.npy', matrix)
print("Saved")
