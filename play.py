from matplotlib import patches
import numpy as np
import matplotlib.pyplot as plt
from utils import *
from keras.datasets import mnist
from sklearn.feature_extraction import image

(train_X, train_y), (test_X, test_y) = mnist.load_data()
img = train_X[0]


patch_shape = (14,14)
stride_len = (5,5)
p, tls = extract_grayscale_patches(
				img= img, 
				shape=patch_shape,
				stride=stride_len)
t, l = tls
fig, ax = plt.subplots()
for i in range(len(t)):
	ax.add_patch(patches.Rectangle((t[i], l[i]), patch_shape[0], patch_shape[1], linewidth=1, edgecolor='r', facecolor='none'))
recon, _ = reconstruct_from_grayscale_patches(p, tls)
ax.imshow(recon)
plt.show()
