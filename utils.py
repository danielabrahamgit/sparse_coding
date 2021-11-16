import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from utils import *

# 2DFT and inverse 
def fft2c(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

def ifft2c(F):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))

def gen_noisy(img_filename, peak=255):
	pois = np.array(ImageOps.grayscale(Image.open(img_filename)))
	pois_noisy = pois * (peak / np.max(pois))
	img_noisy = np.zeros(pois.shape)
	for r in range(img_noisy.shape[0]):
		for c in range(img_noisy.shape[1]):
			img_noisy[r,c] = np.random.poisson(pois_noisy[r,c])
	img_noisy = (img_noisy * 255) / np.max(img_noisy)
	img_noisy = img_noisy.astype(np.uint8)
	# plt.hist(img_noisy.flatten(), bins=255)
	plt.imshow(img_noisy, cmap='gray')
	plt.show() 

def get_patches(img, patch_size, strides):
	assert len(patch_size) == 2
	assert len(strides) == 2
	
	patches = []

	# Define demensions
	prows, pcols = patch_size
	stride_r, stride_c = strides
	irows, icols = img.shape

	

	