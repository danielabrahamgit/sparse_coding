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

def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = np.meshgrid( np.arange(shape[1]),np.arange(shape[0]))
    l, t = np.meshgrid(
        np.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        np.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = np.tile( px[None,:,:], (t.size,1,1)) + np.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = np.tile( py[None,:,:], (t.size,1,1)) + np.tile( t[:,None,None], (1,shape[0],shape[1]))
    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t,l)

def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = np.max( origin[1] ) + patch_width
    img_height   = np.max( origin[0] ) + patch_height

    out = np.zeros( (img_height,img_width) )
    wgt = np.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/np.maximum( wgt, epsilon ), wgt
	

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


# Algorithm 5.1 in https://www.learningtheory.org/colt2009/papers/009.pdf
def lsc(Y, D, sigma):

    def omega_bar(t):
        return t ** 2 * (t <= 1) + (2 * np.abs(t) - 1) * (t > 1)

    beta = 4 * sigma * sigma
    tau = 4 * sigma / np.linalg.norm(D, ord='fro')
    T = D.shape[0]
    # h = beta / (np.linalg.norm(D, ord='fro') ** 2)
    h = beta / (D.shape[0] * D.shape[1])

    L = np.zeros(D.shape[1])
    lambd = np.zeros_like(L)
    H = 0

    DD = D.T @ D
    Dy = D.T @ Y
    i = 0
    while H < T:
        i += 1
        nablaV = (2 / beta) * (Dy - DD @ L)
        nablaV = nablaV - 4 * L / (tau ** 2 + L ** 2)
        L = L + h * nablaV + np.sqrt(2 * h) * np.random.normal(0, 1, L.shape)
        H = H + h
        lambd = lambd + h * L / T

    return lambd