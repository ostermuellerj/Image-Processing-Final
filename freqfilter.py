import cv2 as cv
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
from math import sqrt, exp

img = cv.imread('bee.jpg', 0)

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

# GAUSSIAN LOWPASS AND HIGHPASS FILTERS
def dist(p,q):
    return sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

# Generate a gaussion filter given D0, a size, and c=["GLPF","GHPF"]
def gaussianFilter(D0,imgShape,c):
	out = np.zeros(imgShape[:2])
	rows, cols = imgShape[:2]
	center = (rows/2,cols/2)
	for x in range(cols):
		for y in range(rows):
			if c == "GLPF":
				out[y,x] = exp(((-dist((y,x),center)**2)/(2*(D0**2))))
			if c == "GHPF":
				out[y,x] = 1 - exp(((-dist((y,x),center)**2)/(2*(D0**2))))	
	return out

# Show Gaussian filtering in frequency domain, c=["GLPF","GHPF"]
def showGaussian(img, c):
	k = np.fft.fft2(img)
	kshift = np.fft.fftshift(k)
	spectrum = np.log(np.abs(kshift))
	
	# Show input image and spectrum
	plt.subplot(151),plt.imshow(img, cmap = 'gray'), plt.title('Input Image'), plt.axis('off')
	plt.subplot(152),plt.imshow(spectrum, cmap = 'gray'), plt.title('Spectrum'), plt.axis('off')

	LP = kshift*gaussianFilter(50, img.shape, c)
	LP_shift = np.fft.ifftshift(LP)
	LP_img = np.fft.ifft2(LP_shift)

	# Show filter and filtered output
	plt.subplot(153), plt.imshow(gaussianFilter(50, img.shape, c), "gray"), plt.title(c), plt.axis('off')
	plt.subplot(154),plt.imshow(np.log(np.abs(LP)), cmap = 'gray'), plt.title('Filtered Spectrum'), plt.axis('off')
	plt.subplot(155), plt.imshow(np.abs(LP_img), "gray"), plt.title("Filtered Image"), plt.axis('off')
	plt.suptitle('1/5: '+c)
	plt.show()
#showGaussian(img, "GLPF")
#showGaussian(img, "GHPF")

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

# LAPLACIAN OF GAUSSIAN	
# Laplacian of GHPF with D0=100
def showLaplacian(img):
	k = np.fft.fft2(img)
	kshift = np.fft.fftshift(k)
	spectrum = np.log(np.abs(kshift))
	
	# Show input image and spectrum
	plt.subplot(151),plt.imshow(img, cmap = 'gray'), plt.title('Input Image'), plt.axis('off')
	plt.subplot(152),plt.imshow(spectrum, cmap = 'gray'), plt.title('Spectrum'), plt.axis('off')
	
	gaussian = gaussianFilter(100, img.shape, "GHPF")
	LoG_filter = cv.Laplacian(np.float32(gaussian), cv.CV_32F, ksize=5)
	LoG = kshift*LoG_filter
	# LoG = kshift*cv.Laplacian(np.float32(k), cv.CV_32F, ksize=3)
	LoG_shift = np.fft.ifftshift(LoG)
	LoG_img = np.fft.ifft2(LoG_shift)
	
	# Show filter and filtered output
	plt.subplot(153), plt.imshow(np.log(np.abs(LoG_filter)), "gray"), plt.title("LoG Filter"), plt.axis('off')
	plt.subplot(154),plt.imshow(np.log(np.abs(LoG)), 'gray'), plt.title('Filtered Spectrum'), plt.axis('off')
	plt.subplot(155), plt.imshow(np.abs(LoG_img), "gray"), plt.title("Filtered Image"), plt.axis('off')
	plt.suptitle('2/5: "Laplacian of Gaussian"')	
	plt.show()
#showLaplacian(img)

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

# DIFFERENCE OF GAUSSIAN

# Sharpening function using two GHPFs
def showDoG(img, D0_0 = 100, D0_1 = 150):
	k = np.fft.fft2(img)
	kshift = np.fft.fftshift(k)
	spectrum = np.log(np.abs(kshift))
	
	# Show input image and spectrum
	plt.subplot(151),plt.imshow(img, cmap = 'gray'), plt.title('Input Image'), plt.axis('off')
	plt.subplot(152),plt.imshow(spectrum, cmap = 'gray'), plt.title('Spectrum'), plt.axis('off')
	
	G1 = gaussianFilter(D0_0, img.shape, "GHPF")
	G2 = gaussianFilter(D0_1, img.shape, "GHPF")

	DoG_filter = G2-G1
	DoG = kshift*DoG_filter
	DoG_shift = np.fft.ifftshift(DoG)
	DoG_img = np.fft.ifft2(DoG_shift)
	
	# Show filter and filtered output
	plt.subplot(153), plt.imshow(np.log(np.abs(DoG_filter)), "gray"), plt.title("DoG Filter"), plt.axis('off')
	plt.subplot(154),plt.imshow(np.log(np.abs(DoG)), 'gray'), plt.title('Filtered Spectrum'), plt.axis('off')
	plt.subplot(155), plt.imshow(np.abs(DoG_img), "gray"), plt.title("Filtered Image"), plt.axis('off')
	plt.suptitle('3/5: Difference of Gaussian')	
	plt.show()
#showDoG(img)

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

# NOTCH FILTERING
# Generates notch filter at coordinates cy, cx
def notchFilter(D0,imgShape,cx,cy):
	out = np.zeros(imgShape[:2])
	rows, cols = imgShape[:2]
	center = ((rows/2)+cx,(cols/2)+cy)
	for x in range(cols):
		for y in range(rows):
			out[y,x] = 1 - exp(((-dist((y,x),center)**2)/(2*(D0**2))))
	return out

# Combine and test notch filters for "car.jpg" (image-specific solution)
def showNotch():
	img = cv.imread('car.jpg', 0)
	k = np.fft.fft2(img)
	kshift = np.fft.fftshift(k)
	spectrum = np.log(np.abs(kshift))

	# Show input image and spectrum
	plt.subplot(161),plt.imshow(img, cmap = 'gray'), plt.title('Input Image'), plt.axis('off')
	plt.subplot(162),plt.imshow(spectrum, cmap = 'gray'), plt.title('Spectrum'), plt.axis('off')

	# Location of peaks in spectrum map
	points = [(-40, -30), (43, -30), (-80, -30), (86, -30), (-120, -30), (120, -30)]

	# For each peak, add a highpass at (x, y) and (-x,-y)
	# The size of each peak tapers based on y
	# These parameters are tuned specically based on the input "car.jpg" image 
	notch_filter = notchFilter(1, img.shape, -999, -999)
	h = img.shape[1]
	for i in points:
		notch_filter *= notchFilter((h-np.abs(i[0]))/10, img.shape, i[0], i[1])
		notch_filter *= notchFilter((h-np.abs(i[0]))/10, img.shape, -i[0], -i[1])
	notch = kshift*notch_filter
	notch_shift = np.fft.ifftshift(notch)
	notch_img = np.fft.ifft2(notch_shift)

	# Add extra GLPF
	notch2 = kshift*notch_filter*gaussianFilter(50, img.shape, "GLPF")
	notch_shift2 = np.fft.ifftshift(notch2)
	notch_img2 = np.fft.ifft2(notch_shift2)

	# Show filter and filtered output
	plt.subplot(163), plt.imshow(notch_filter, "gray"), plt.title("Notch Filter"), plt.axis('off')
	plt.subplot(164),plt.imshow(np.log(np.abs(notch)), cmap = 'gray'), plt.title('Filtered Spectrum'), plt.axis('off')
	plt.subplot(165), plt.imshow(np.abs(notch_img), "gray"), plt.title("Filtered Image"), plt.axis('off')
	plt.subplot(166), plt.imshow(np.abs(notch_img2), "gray"), plt.title("Filtered Image + GLPF"), plt.axis('off')
	plt.suptitle('4/5: Notch Filtering')
	plt.show()
#showNotch()

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

# SPATIAL KERNEL TO FREQUENCY FILTER
# Given an image (of any size) and a kernel (of any size smaller than the image), this function 
# demonstrates the effects of using the kernal as a convolutional filter in the spatial domain 
# vs using an element-wise multiplication in the frequency domain. The function also outputs the 
# difference between these two results.
def spatialToFrequencyFilter(img, kernel, name=""):
	img = cv.imread('bee_crop.jpg', 0)
	plt.subplot(331), plt.imshow(img, "gray"), plt.title("f(x,y)"), plt.axis('off')
	z = (img.shape[0]*2, img.shape[1]*2)	
	z = np.zeros(z, complex)	
	z1 = np.copy(z)
	
	z[:img.shape[0],:img.shape[1]] = img
	
	# ~ # Center fourier on frequency rectangle
	for x in range(img.shape[0]):
		for y in range(img.shape[1]):
			z[x][y] *= -1**(x+y)
	
	plt.subplot(332), plt.imshow(np.float32(abs(z)), "gray"), plt.title("Pad + f(x,y)*-1^(x+y)"), plt.axis('off')
	
	# Image spectrum
	k = np.fft.fft2(z).astype(complex)
	kshift = np.fft.fftshift(k).astype(complex)
	spectrum = np.log(np.abs(kshift))
	#plt.subplot(333), plt.imshow(spectrum, "gray"), plt.title("img"), plt.axis('off')	
	
	
	h, w = z.shape
	hi, wi = kernel.shape
	# Pad kernel with zeros
	z1[int((h/2)-(hi/2)):int((h/2)+(hi/2)),int((w/2)-(wi/2)):int((w/2)+(wi/2))] = kernel
	
	# Kernel Spectrum
	k1 = np.fft.fft2(z1).astype(complex)
	k1shift = np.fft.fftshift(k1).astype(complex)

	plt.subplot(333), plt.imshow(np.log(np.abs(kshift)), "gray"), plt.title("F(u,v)"), plt.axis('off')	
	#plt.subplot(334), plt.imshow(np.float32(z1), "gray"), plt.title("filter"), plt.axis('off')	
	plt.subplot(334), plt.imshow(np.float32(z1[int((h/2)-(hi/2)):int((h/2)+(hi/2)),int((w/2)-(wi/2)):int((w/2)+(wi/2))]), "gray"), plt.title("H(u,v) (zoomed)"), plt.axis('off')	
	plt.subplot(335), plt.imshow(np.log(np.abs(k1shift)), "gray"), plt.title("H(u,v)"), plt.axis('off')	
			
	# Combination of img spectrum and kernel spectrum
	f = np.array(k1shift*kshift).astype(complex)
	plt.subplot(336), plt.imshow(np.log(np.abs(f)), "gray"), plt.title("F(u,v)H(u,v)"), plt.axis('off')	
	f_shift = np.fft.ifftshift(f)
	
	# Output of ifft, desired result is in bottom right quadrant
	f_img = np.fft.ifft2(f_shift).astype(complex)
	for x in range(f_img.shape[0]):
		for y in range(f_img.shape[1]):
			f_img[x][y] *= -1**(x+y)
	#plt.subplot(333), plt.imshow((np.abs(f_img)), "gray"), plt.title("f_img"), plt.axis('off')	
	final = f_img[int(h/2):,int(w/2):]	
	
	kernel = kernel/(np.sum(kernel) if np.sum(kernel)!=0 else 1)
	convolve = cv.filter2D(img,-1,kernel)
	
	diff = (img.shape[0], img.shape[1])
	diff = np.zeros(diff, np.int8)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			diff[i][j] = np.abs(convolve[i][j] - final[i][j])
	
	# ~ diff = cv.absdiff(np.float32(img), np.float32(final))
	plt.subplot(337), plt.imshow(np.float32(np.abs(final)), "gray"), plt.title("Final Result"), plt.axis('off')
	plt.subplot(338), plt.imshow(convolve, "gray"), plt.title("Spatial Convolve"), plt.axis('off')
	plt.subplot(339), plt.imshow(np.float32(np.abs(diff)), "gray"), plt.title("Spatial vs Freq diff"), plt.axis('off')
	plt.suptitle('5/5: 3X3 Kernel: Convt. vs Freq Filtering'+name) 
	plt.show()
# Simple sobel-x kernel
#S = np.array(
#[[-2, 0, 2],
#[-4, 0, 4],
#[-2, 0, 2]])
#spatialToFrequencyFilter(img, S, " (Sobel)")

# Simple gaussian
#G = np.array(
#[[1, 2, 1],
#[2, 4, 2],
#[1, 2, 1]])
#spatialToFrequencyFilter(img, G, " (Gaussian)")

# --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- # --------- #

def main():
	# Gaussion HP and LP:
	showGaussian(img, "GLPF")
	showGaussian(img, "GHPF")

	# Laplacian of Gaussian:
	showLaplacian(img)
	
	# Difference of Gaussian:
	showDoG(img)
	
	# Simple notch filter:
	showNotch()
	
	# Simple sobel-x kernel: spatial and frequency filtering
	S = np.array(
	[[-2, 0, 2],
	[-4, 0, 4],
	[-2, 0, 2]])
	spatialToFrequencyFilter(img, S, " (Sobel)")

	# Simple gaussian kernel: spatial and frequency filtering
	G = np.array(
	[[1, 2, 1],
	[2, 4, 2],
	[1, 2, 1]])
	spatialToFrequencyFilter(img, G, " (Gaussian)")

main()
