"""
Step 1 of the pipeline: extract one 128-d identity embedding per image and
save the embeddings, labels and source paths as numpy arrays.

Usage (this repo's default paths):
    python export_embeddings.py
Reads every image under ./data/lfw (one folder per person), detects a face
in each image, and writes:
    np_embeddings/embeddings.npy      (N, 128) float32 embeddings
    np_embeddings/label_list.npy      (N,)    integer person index
    np_embeddings/label_strings.npy   (N,)    person name (folder name)
    np_embeddings/image_list.npy      (N,)    source path of each image
All four arrays share exactly the same row order.

Theory
------
Model: the default ./models/20170512-110547 checkpoint is an
Inception-ResNet-V1 (the architecture of the original FaceNet paper,
Schroeder et al. 2015) trained with softmax cross-entropy against a
44,052-class identity head. Its output is a 128-d embedding that is
L2-normalized, i.e. every face is a point on the unit sphere. That is the
property the rest of the pipeline relies on: for unit vectors the Euclidean
distance is a monotonic function of the cosine angle, so step 2 can rank
image pairs by Euclidean distance without computing cosines.

Input preprocessing must reproduce what the model was trained on, namely
160x160 crops in which the face is centered. This script approximates the
official FaceNet alignment:
  1. detect the face with MTCNN (the P-Net/R-Net/O-Net cascade of the
     original align/ code; weights in src/align/det1..3.npy),
  2. expand the detection box by --margin/2 on each side (44 px total,
     the official margin) and crop,
  3. resize bilinearly to --image_size (160x160),
  4. prewhiten: subtract the image mean and divide by its std, which
     normalises away per-image lighting and contrast differences.
Feeding unaligned images (a whole photo, or an un-centered face) moves the
embedding noticeably -- keep the margin/size defaults. Images with no face
detected fall back to a centre crop so that one bad image cannot abort a
multi-hour run (this never triggered on LFW).

Caveats
-------
* Row order: persons appear alphabetically, but images *within* a person
  come from os.listdir() order (filesystem readdir order), which is NOT
  reproducible across runs or machines. Never index embeddings by position;
  always join the arrays through image_list.npy (full paths).
* --is_aligned is a vestigial flag from the original script (type=str,
  compared with `is True`, so it is false in every case): this fork ALWAYS
  runs the MTCNN detection branch above. There is no aligned-input fast
  path; aligned crops would simply be re-detected.
* Embeddings are deterministic for a given image (inference mode, fixed
  weights), but different builds of the detection stack can shift the crop
  by a pixel or two, which shows up as small embedding differences (full
  LFW re-run: median cosine 0.998, 61/13,233 images cos < 0.9, worst case
  ~-0.33 where the detector latched onto a different face).

I started with compare.py from David Sandberg, and modified it to export
the embeddings. The image loading is done using the facenet library if the
image is pre-aligned. If the image isn't pre-aligned, I use the compare.py
function. I've found working with the embeddings useful for classification
models.

Charles Jekel 2017

"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import cv2
import tensorflow as tf
import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face

from six.moves import xrange

# Legacy TensorFlow 1.x graph API: the functions below use the TF 1.x
# session/graph/variable API, which is only available under tf.compat.v1
# in TensorFlow 2.x (and requires eager execution to be disabled).
tf.compat.v1.disable_eager_execution()

def main(args):
    # Walk --data_dir: every immediate subfolder is one person (one class).
    # Folders are sorted alphabetically, but images *within* a folder come
    # from os.listdir() (unsorted) -- this is where the non-reproducible
    # row order of the output arrays comes from; see the module docstring.
    train_set = facenet.get_dataset(args.data_dir)
    # image_list: full source path of each image
    # label_list: integer class index of its containing folder
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # Person names (folder names) as strings, alongside the integer labels;
    # steps 2-3 do not consume them, they are saved for human reference.
    label_strings = []
    for i in range(len(image_list)):
        head, tail = os.path.split(os.path.split(image_list[i])[0])
        label_strings.append(tail)

    with tf.compat.v1.Graph().as_default():

        with tf.compat.v1.Session() as sess:

            # Load the model: restore the Inception-ResNet-V1 weights from
            # the meta/ckpt files in --model_dir into the default graph.
            facenet.load_model(args.model_dir)

            # Get input and output tensors of that graph: 160x160x3 RGB
            # floats in, 128-d unit-sphere embeddings out.
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = args.image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches -1:
                    n = nrof_images
                else:
                    n = i*batch_size + batch_size
                # Get images for the batch.
                # --is_aligned is a vestigial str flag that can never be
                # `True` here, so this always runs the detection/alignment
                # path (see the module docstring).
                if args.is_aligned is True:
                    images = facenet.load_data(image_list[i*batch_size:n], False, False, args.image_size)
                else:
                    images = load_and_align_data(image_list[i*batch_size:n], args.image_size, args.margin, args.gpu_memory_fraction)
                # phase_train=False switches the BatchNorm layers to their
                # frozen running statistics (inference mode) instead of the
                # per-batch statistics used while training.
                feed_dict = { images_placeholder: images, phase_train_placeholder:False }
                # One forward pass through the Inception-ResNet-V1:
                # (n, 160, 160, 3) batch in, (n, 128) embeddings out.
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i*batch_size:n, :] = embed
                print('Completed batch', i+1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            # Export embeddings and labels. All four arrays share the row
            # order of image_list (see module docstring); consumers must
            # join them by full path, never by row position.
            label_list  = np.array(label_list)

            np.save(args.embeddings_name, emb_array)
            np.save('./np_embeddings/label_list.npy', label_list)
            label_strings = np.asarray(label_strings)
            np.save('./np_embeddings/label_strings.npy', label_strings)
            np.save('./np_embeddings/image_list.npy', image_list)

def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    """
    Detect a face in each image with MTCNN and return the training-format
    batch: (n, image_size, image_size, 3) float32 array of centred,
    prewhitened face crops -- exactly what the model's input tensor expects.
    """

    # MTCNN (Multi-task Convolutional Neural Network) detection settings --
    # the defaults of the official FaceNet align tool. MTCNN is a 3-stage
    # cascade: P-Net proposes candidate boxes over a multi-scale image
    # pyramid (downsampled by `factor` per level), R-Net re-scores them,
    # O-Net refines the box and predicts landmarks; each stage drops
    # detections below its confidence `threshold`. `minsize` is the
    # smallest face the pyramid attempts to find (most LFW faces are
    # well above 20 px, so it is a speed floor rather than a recall limit).
    minsize = 20 # minimum face size in pixels
    threshold = [ 0.6, 0.7, 0.7 ]  # per-stage confidence cutoffs (P, R, O)
    factor = 0.709 # image pyramid scale per level (~1/sqrt(2))

    # Build the 3-stage cascade in-graph and load its pretrained weights
    # (src/align/det{1,2,3}.npy). Done once, then reused for every image.
    print('Creating networks and loading parameters')
    with tf.compat.v1.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        # OpenCV decodes to BGR; the model was trained on RGB.
        img = cv2.imread(os.path.expanduser(image_paths[i]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        if bounding_boxes.shape[0] == 0:
            # Defensive fallback: no face detected. Crop the central square
            # instead of aborting the whole (multi-hour) run on one bad image.
            side = int(min(img_size))
            cx, cy = img_size[1] // 2, img_size[0] // 2
            half = side // 2
            bb = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.int32)
        else:
            det = np.squeeze(bounding_boxes[0,0:4])
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img_size[1])
            bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        # Reproduce the training crop: detected box + margin (44 px total)
        # -> bilinear resize to 160x160 (the size the model was trained
        # on) -> prewhiten (per-image mean/std) to normalise lighting.
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = cv2.resize(cropped, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    # Single (n, 160, 160, 3) tensor: the model's `input` batch.
    images = np.stack(img_list)
    return images

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
        help='Directory containing the meta_file and ckpt_file', default='./models/20170512-110547')
    parser.add_argument('--data_dir', type=str,
        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.',
                        default='./data/lfw')
    parser.add_argument('--is_aligned', type=str,
        help='Is the data directory already aligned and cropped?', default=False)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
        help='Margin for the crop around the bounding box (height, width) in pixels.',
        default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.',
        default=1.0)
    parser.add_argument('--image_batch', type=int,
        help='Number of images stored in memory at a time. Default 500.',
        default=500)

    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str,
        help='Enter string of which the embeddings numpy array is saved as.',
        default='./np_embeddings/embeddings.npy')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
