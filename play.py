from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from keras.datasets import mnist
from sklearn.feature_extraction import image

# Parameters
# --------------------------------------
patch_shape = (5,5)
stride_len = (1,1)
# --------------------------------------

# Gen training/test data
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# Use patches instead of full images
train_patches = []
test_patches = []
for img in train_X:
	p, tls = extract_grayscale_patches(
					img=img, 
					shape=patch_shape,
					stride=stride_len)
	train_patches.append(p)

for img in test_X:
	p, tls = extract_grayscale_patches(
					img=img, 
					shape=patch_shape,
					stride=stride_len)
	test_patches.append(p)
	
train_patches = np.array(train_patches)
test_patches = np.array(test_patches)

recon, _ = reconstruct_from_grayscale_patches(p, tls)

