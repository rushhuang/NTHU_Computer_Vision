import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def imageshow(img, cmap='bgr', savefile=False, filename='output.jpg'):
	""" 
	Print image or Save it.

	img: Image to show or to save.
	cmap: Color map of img. Default to 'bgr'
	savefile: To save image or not. Default to 'False'.
	filename: The filename you want the output image has.
			  Default to 'output.jpg'
	"""
	cwd = os.getcwd() + '/'
	des = cwd + filename

	if savefile:
		cv2.imwrite(des, img)
	else:
		if(cmap == 'bgr'):
			# OpenCV considers float only when values range
			# from 0-1.
			img = img.astype('uint8')
			# Change order from bgr to rgb for plt to show
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			plt.imshow(img)
		else:
			plt.imshow(img, cmap=cmap)
		
		plt.show()

def GetExtendImage(img, kernel_size=3):
	"""
	Get image having 3 channel with half of kernel_size as padding.

	img: Image with **3 channel** to extend.
	kernel_size: Decide how many pixel to extend as padding.
				 Default to 3.
	"""
	# Get input image size
	y = img.shape[0]
	x = img.shape[1]

	# Get half of the kernel, which means the
	# pixel number to extend.
	half = int(kernel_size/2)

	# Initialize tmp array
	tmp_len_x = x+2*half
	tmp_len_y = y+2*half
	tmp = np.zeros((tmp_len_y, tmp_len_x, 3))
	
	# Extending along x axis of original pic
	i_end = tmp_len_y-half-1
	for i in range(tmp_len_y):
		i_half = i-half
		if i_half < 0:
			pass
		elif i > i_end:
			pass
		else:
			for itr in range(tmp_len_x):
				itr_half = itr-half
				if itr_half < 0:
					tmp[i][itr][0] = img[i_half][0][0]
					tmp[i][itr][1] = img[i_half][0][1]
					tmp[i][itr][2] = img[i_half][0][2]
				elif itr > tmp_len_x-half-1:
					tmp[i][itr][0] = img[i_half][-1][0]
					tmp[i][itr][1] = img[i_half][-1][1]
					tmp[i][itr][2] = img[i_half][-1][2]
				else:
					tmp[i][itr][0] = img[i_half][itr_half][0]
					tmp[i][itr][1] = img[i_half][itr_half][1]
					tmp[i][itr][2] = img[i_half][itr_half][2]

	# Transpose both image on dimension 1 and 2 for simplicity
	tmp = np.transpose(tmp, (1, 0, 2))
	img = np.transpose(img, (1, 0, 2))
	
	# Extending along y axis of original pic
	for i in range(tmp_len_x):
		i_half = i-half
		for itr in range(tmp_len_y):
			itr_half = itr-half
			tmp_len_y_half1 = tmp_len_y-half-1
			if i_half < 0:
				if itr_half < 0:
					tmp[i][itr][0] = tmp[i][half][0]
					tmp[i][itr][1] = tmp[i][half][1]
					tmp[i][itr][2] = tmp[i][half][2]
				elif itr > tmp_len_y_half1:
					tmp[i][itr][0] = tmp[i][tmp_len_y_half1][0]
					tmp[i][itr][1] = tmp[i][tmp_len_y_half1][1]
					tmp[i][itr][2] = tmp[i][tmp_len_y_half1][2]
				else:
					pass
			elif i > tmp_len_x-half-1:
				if itr_half < 0:
					tmp[i][itr][0] = tmp[i][half][0]
					tmp[i][itr][1] = tmp[i][half][1]
					tmp[i][itr][2] = tmp[i][half][2]
				elif itr > tmp_len_y_half1:
					tmp[i][itr][0] = tmp[i][tmp_len_y_half1][0]
					tmp[i][itr][1] = tmp[i][tmp_len_y_half1][1]
					tmp[i][itr][2] = tmp[i][tmp_len_y_half1][2]
				else:
					pass
			else:
				if itr_half < 0:
					tmp[i][itr][0] = img[i_half][0][0]
					tmp[i][itr][1] = img[i_half][0][1]
					tmp[i][itr][2] = img[i_half][0][2]
				elif itr > tmp_len_y-half-1:
					tmp[i][itr][0] = img[i_half][-1][0]
					tmp[i][itr][1] = img[i_half][-1][1]
					tmp[i][itr][2] = img[i_half][-1][2]
				else:
					tmp[i][itr][0] = img[i_half][itr_half][0]
					tmp[i][itr][1] = img[i_half][itr_half][1]
					tmp[i][itr][2] = img[i_half][itr_half][2]

	# Back to original
	tmp = np.transpose(tmp, (1, 0, 2))

	# Display the result
	# imageshow(tmp, savefile=False, filename='extended.jpg')

	print("Done extension...")
	return tmp
	### End of extending the image. ###

def gaussian_smooth(img, sigma=5, kernel_size=10):
	'''
	Return image with Gaussian smoothing.

	img: Image with **3 channels**.
	sigma: Sigma value to do Gaussian smoothing.
		   Default to 5.
	kernel_size: Kernel size to do Gaussian smoothing.
				 Default to 10.
	'''
	
	### Calculating the Gaussian kernel.###
	# Get the half point of kernel, N
	half = int(kernel_size/2)

	# Initialize kernel
	kernel = []
	for i in range(2*half+1):
		kernel.append(i)
	kernel = np.array(kernel, dtype='float')

	# Initialize the central element
	kernel[half] = 1.

	# Calculate G"_n
	for i in range(1, half+1):
		x_n = 3. * i / half
		kernel[half - i] = kernel[half + i] = ( (1/(sigma * np.sqrt(2.*np.pi)))
												* np.exp(-(i**2)/(2.*(sigma**2))) )

	# Calculate k'
	k = sum(kernel)

	# G'_n = G"_n / k'
	kernel /= k

	print(kernel)
	### End of calculating the Gaussian kernel.###


	### Extending the image for later sliding. ###
	tmp = GetExtendImage(img, kernel_size=kernel_size)
	tmp_len_x = tmp.shape[1]
	tmp_len_y = tmp.shape[0]
	### End of extending the image. ###


	### Comvolving with the kernel. ###
	# Get image size
	y = img.shape[0]
	x = img.shape[1]

	# Reshape array: [H, W, Channel] to [Channel, H, W]
	tmp = np.transpose(tmp, (2, 0, 1))
	img = np.transpose(img, (2, 0, 1))

	# Convolving along x axis
	for i in range(tmp_len_y):
		for j in range(x):
			j_end = j+2*half+1
			tmp[0][i][j] = np.dot(tmp[0][i][j:j_end], kernel)
			tmp[1][i][j] = np.dot(tmp[1][i][j:j_end], kernel)
			tmp[2][i][j] = np.dot(tmp[2][i][j:j_end], kernel)

	# Transpose to apply kernel along y axis
	tmp = np.transpose(tmp, (0, 2, 1))

	# Transpose img to align to tmp
	img = np.transpose(img, (0, 2, 1))

	# Convolving along y axis
	i_end = tmp_len_x-2*half
	for i in range(0, i_end):
		for j in range(y):
			j_end = j+2*half+1
			img[0][i][j] = np.dot(tmp[0][i][j:j_end], kernel)
			img[1][i][j] = np.dot(tmp[1][i][j:j_end], kernel)
			img[2][i][j] = np.dot(tmp[2][i][j:j_end], kernel)
	
	tmp = np.transpose(tmp, (1, 2, 0))
	img = np.transpose(img, (1, 2, 0))

	# !!!Code below would yield wrong output.
	# The output image would have extended pixel at the right side.
	# Unlike what I thought, which is righthand side of the original
	# image would be on the top of the transposed image.
	#
	# for i in range(2*half+1, tmp_len_x):
	# 	for j in range(y):
	# 		img[i-2*half-1][j] = np.dot(tmp[i][j:j+2*half+1], kernel)

	# Transpose back to normal
	img = np.transpose(img, (1, 0, 2))

	###End of convolving with the kernel. ###

	# Display the result
	# imageshow(img, savefile=False, filename='gaussian_kernel_5.jpg')

	print("Done Gaussian...")
	return img

def sobel_edge_detection(img):
	'''
	Detect edge of the image, and return image derivatives along x
	and y axis.

	img: Image with **3 channels**.
	'''

	# Set the threshold, '70' based on wiki
	threshold = 100

	# Get image size
	y = img.shape[0]
	x = img.shape[1]

	# For magnitude map
	Mag_map = np.zeros((y, x))

	# For direction map, which is colorized
	Dir_map = np.zeros((y, x, 3))

	# For Image derivatives along x and y
	Ix = np.zeros((y, x))
	Iy = np.zeros((y, x))

	# Define Sobel operator
	Gx = np.array([[-1, 0, 1],
				   [-2, 0, 2],
				   [-1, 0, 1]])
	
	Gy = np.array([[-1, -2, -1],
				   [0, 0, 0],
				   [1, 2, 1]])

	# Define direction color
	Red = np.array([0,0,255])
	Yellow = np.array([0,255,255])
	Green = np.array([0,255,0])
	Cyan = np.array([255,255,0])


	### Extending the image for later sliding. ###
	tmp = GetExtendImage(img, kernel_size=3)
	tmp_len_x = tmp.shape[1]
	tmp_len_y = tmp.shape[0]
	### End of extending the image. ###

	# From BGR to Gray
	tmp = tmp.astype('uint8')
	tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

	### Convolving with the kernel. ###
	for i in range(tmp_len_y-2):
		i_end = i+3
		for j in range(tmp_len_x-2):
			j_end = j+3
			Ix[i][j] = np.sum(np.sum(tmp[i:i_end, j:j_end] * Gx))
			Iy[i][j] = np.sum(np.sum(tmp[i:i_end, j:j_end] * Gy))

			## Calculating Gradient Magnitude
			# G = sqrt(Gx**2 + Gy**2)
			gradient_mag = np.sqrt(Ix[i][j]**2 + Iy[i][j]**2)

			# max(Mag_map, threshold)
			if (gradient_mag > threshold):
				Mag_map[i][j] = gradient_mag


				## Calculating Gradient Direction
				# Only apply on 'detected edges'
				# Theta = arctan(Gy/Gx)
				angle = np.arctan( Iy[i][j] / Ix[i][j] )

				# arctan(1) ~ 0.79
				if(angle > 0.79):
					Dir_map[i][j] = Red
				elif(angle >= 0 and angle < 0.79):
					Dir_map[i][j] = Yellow
				elif(angle >=-0.79 and angle < 0):
					Dir_map[i][j] = Green
				elif(angle < -0.79):
					Dir_map[i][j] = Cyan
				## End of Calculating Gradient Direction

			else:
				Mag_map[i][j] = 0

			## End of Calculating Gradient Magnitude
	### End of Convolving with the kernel. ###

	# Display the result
	# imageshow(Ix, cmap='gray', savefile=False,
	#				filename='sobel_Ix_kernel_10.jpg')
	# imageshow(Iy, cmap='gray', savefile=False,
	#				filename='sobel_Iy_kernel_10.jpg')
	# imageshow(Mag_map, cmap='gray', savefile=False,
	#				filename='sobel_mag_kernel_10.jpg')
	# imageshow(Dir_map, savefile=False,
	#				filename='sobel_dir_kernel_10.jpg')

	print("Done sobel...")
	return Ix, Iy
	

def structure_tensor(img, img_mix, k=0.04):
	'''
	To calculate structure tensor of input image.
	Return Harris response and tha maximum of Harris response.

	img: Image with **3 channels**. To see the results.
	img_mix: **3 channels** which are [Ix, Iy, img_gray]
		- Ix: Derivative along x axis from sobel edge detection.
		- Iy: Derivative along y axis from sobel edge detection.
		- img_gray: Original image in gray scale.
	k: Constant k in the formula to calculate Harris response.
	   In range of 0.04~0.06. Default to 0.04. 
	'''

	# Get image size
	y = img.shape[0]
	x = img.shape[1]

	# For window size to calculate structure tensor
	window_size = 3
	half = int(window_size/2)

	# For structure tensor, M
	M00 = np.zeros((y, x))
	M11 = np.zeros((y, x))
	M01 = np.zeros((y, x)) # M01 = M10

	# Extract the img_gray before feeding img_mix into GetExtendImage(),
	# for later marking(if you want).
	img_gray = img_mix[2]

	### Extending the image for later sliding. ###
	tmp = GetExtendImage(img_mix, kernel_size=window_size)
	tmp_len_x = tmp.shape[1]
	tmp_len_y = tmp.shape[0]
	### End of extending the image. ###

	# Extract Ix, Iy from img_mix
	# (H, W, 3) to (3, H, W)
	img_mix = np.transpose(tmp, (2, 0, 1))
	Ix = img_mix[0]
	Iy = img_mix[1]

	### Convolving. Calculating the structure tensor. ###
	for i in range(y):
		i_end = i+window_size
		for j in range(x):
			j_end = j+window_size

			# NOT Ix[i:i_end][j:j_end]!!!
			# The result is SO DIFFERENt!!!
			M00[i][j] = np.sum((Ix[i:i_end, j:j_end] *
								Ix[i:i_end, j:j_end]))

			M11[i][j] = np.sum((Iy[i:i_end, j:j_end] *
								Iy[i:i_end, j:j_end]))

			M01[i][j] = np.sum((Ix[i:i_end, j:j_end] *
								Iy[i:i_end, j:j_end]))
	### End of convolving. Calculating the structure tensor. ###

	# Determinant and trace of M
	detM = ((M00 * M11) - (M01**2))
	traceM = M00 + M11

	### Calculating Harris response. ###
	harris_response = (detM - (k*(traceM**2)))
	### End of calculating Harris response. ###

	# Find maximum in harris response
	hmax = np.max(harris_response)
	
	# Mark the points on img
	img[harris_response > 0.01*hmax] = [0, 0, 255]

	# Display the result for colored image.
	imageshow(img, cmap='bgr', savefile=True,
					filename='harris_r_kernel_10_3.jpg')
	

	# !!!Method offer by 'https://docs.opencv.org/master/d4/d70/
	# tutorial_anisotropic_image_segmentation_by_a_gst.html'
	# Would yield the same result.
	### Calculating eigenvalue. ###
	# lambda1 = M00 + M11 + sqrt((M00-M11)**2 + 4*M01**2)
	# lambda2 = M00 + M11 - sqrt((M00-M11)**2 + 4*M01**2)
	# t1 = M00 + M11
	# t2 = np.sqrt(((M00 - M11)**2) + (4*(M01**2)))
	# lambda1 = t1 + t2
	# lambda2 = t1 - t2
	### End of calculating eigenvalue. ###
	### Calculating Harris response. ###
	# harris_response = (lambda1*lambda2)-(k*((lambda1+lambda2)**2))
	### End of calculating Harris response. ###

	print("Done Harris...")
	return harris_response, hmax

def nms(img_gray, harris_response, hmax, window_size=3):
	'''
	To do non maximum suppression to input image.

	img_gray: Original image in gray scale.
	harris_response: Harris response from structure_tensor().
	hmax: The maximum value in harris_response from structure_tensor().
	window_size: Window size to do NMS. Default to 3.
	'''
	
	# Get image size
	y = img_gray.shape[0]
	x = img_gray.shape[1]

	# Greate 3 channels for gray image
	img = np.array([[[s, s, s] for s in pixel] for pixel in img_gray])

	# Set up threshold
	threshold = 0.01*hmax

	for i in range(y):
		# Define local box height
		box_y = i+window_size
		if(box_y > y):
			box_y = y

		for j in range(x):

			# Filter those non candidate
			if (harris_response[i][j] < threshold):
				continue

			else:
				# Define local box width
				box_x = j+window_size
				if(box_x > x):
					box_x = x
				
				# Scanning box
				box = harris_response[i:box_y, j:box_x]
				# print(box)

				# Find the max index
				box_max_idx = np.argmax(box)

				# Find the max coordinate
				target = np.unravel_index(box_max_idx, box.shape)

				# Find the max value
				box_max = box[target[0]][target[1]]

				# Non max suppression
				harris_response[i:box_y, j:box_x] = 0
				harris_response[i+target[0]][j+target[1]] = box_max

				# print(harris_response[i:box_y, j:box_x])

	img[harris_response > threshold] = [0, 0, 255]

	# Display the result
	imageshow(img, cmap='bgr', savefile=False,
					filename='results/revised_nms_window_3_scale05.jpg')

	print("Finish!")


def Rotate(img, angle=30):
	"""
	Rotate the image by given degree.

	img: The image you want to rotate.
	angle: Degree you want the image to be rotated.
		   Default to '30'.
	"""

	img_center = tuple(np.array(img.shape[1::-1]) / 2)
	rotate_matrix = cv2.getRotationMatrix2D(img_center, angle, 1.0)
	img = cv2.warpAffine(img, rotate_matrix,
							img.shape[1::-1], flags=cv2.INTER_LINEAR)
	
	print("Done rotation...")
	
	return img


def Scale(img, scale=0.5):
	"""
	Scale the image by given times.

	img: The image you want to scale.
	scale: By how many times you want the image to be scaled.
	"""

	y = img.shape[0]
	x = img.shape[1]

	y_start = int(y/2)
	x_start = int(x/2)
	
	img = cv2.resize(img, (0,0), fx=scale, fy=scale)
	
	if(scale > 1):
		img = img[y_start:y_start+y, x_start:x_start+x]
	
	print("Done scaling...")
	
	return img


if __name__ == '__main__':

	cwd = os.getcwd() + '/'
	image_name = 'original.jpg'
	image_path = cwd + image_name

	# Import the image.
	img = cv2.imread(image_path)

	### Final part ###

	## Rotate the image by 30 degree
	# (Uncomment codes below to do rotation)
	# img = Rotate(img, angle=30)

	## Scale the image by 0.5x
	# (Uncomment codes below to do scaling)
	# img = Scale(img, scale=0.5)

	### End of final part ###

	# Do Gaussian smooth
	img_gaussian = gaussian_smooth(img, kernel_size=10)

	# From BGR to Grayscale
	img_gray = img_gaussian.astype('uint8')
	img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

	# Do Sobel Edge Detection and get gradient magnitude
	Ix, Iy = sobel_edge_detection(img_gaussian)

	# Put Ix, Iy, img_gray into img_mix to go further extension.
	# (3, H, W) to (H, W, 3)
	img_mix = np.array([Ix, Iy, img_gray])
	img_mix = np.transpose(img_mix, (1, 2, 0))

	# Calculate Harris response and the max of Harris response
	harris_response, hmax = structure_tensor(img, img_mix)

	# Do non maximum suppression
	# (Result image in grayscale)
	nms(img_gray, harris_response, hmax, window_size=3)
	# (Result image in color)
	# nms(img, harris_response, hmax, window_size=30)
