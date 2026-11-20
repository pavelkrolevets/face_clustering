"""
Step 2 of the pipeline: all-pairs pairwise distances between embeddings.

Theory
------
FaceNet L2-normalizes every embedding, so each one is a point on the unit
sphere. For two unit vectors a and b:

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b = 2 - 2*cos(theta)

so the *ranking* of pairs by Euclidean distance is exactly the ranking by
cosine similarity. The 128-d Euclidean distance is therefore a faithful
measure of identity dissimilarity: identical faces give ~0, clearly
different people give ~1.4-1.8 (the theoretical range is 0..2, with 2
only for perfectly antipodal embeddings).

The result is an N x N matrix that is symmetric with a zero diagonal.
It is dense, so memory grows as O(N^2) ~ 1.4 GB for the full LFW set
(N = 13,233, float64); on a 48 GB machine this is comfortable, but it is
the step that will hit the wall first on much larger datasets.

Output: ./np_embeddings/matrix.npy  (rows/columns in `image_list.npy` order)
"""
import numpy as np

embeddings = np.load('./np_embeddings/embeddings.npy') # load embeddings of faces
nrof_images = len(embeddings)
matrix = np.zeros((nrof_images, nrof_images))

print('')
# Legacy cosmetic from the original script: print a ruler of all column
# indices, then (every 1000 rows below) one distance value as a rough
# progress marker. At LFW size the ruler alone is a several-megabyte
# line of digits, so expect a big blob here; it is harmless output.
print('')
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
# ~1.4 GB dense float64 for the full LFW set; stored in `names` order.
np.save('./np_embeddings/matrix.npy', matrix)
print("Saved")
