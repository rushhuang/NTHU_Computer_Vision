import cv2
import numpy as np

def GetHomogeneousPoints(point_2d):
	tmp = np.ones((point_2d.shape[0], 1))
	homo = np.concatenate((point_2d, tmp), axis = 1)
	return homo

def GetHomography(a, b):
	# a = Hb
	# b = H^{-1}a

	# Concatenate the equations
	# Ai = [[X1, Y1, 1, 0, 0, 0, −x1X1, −x1Y1, −x1],
	#       [0, 0, 0, X1, Y1, 1, −y1X1, −y1Y1, −y1],...]
	# A is a (8, 9) matrix
	A = np.zeros((8, 9))
	tmp = [9]
	for i in range(b.shape[0]):
		tmp[0:3] = b[i]
		tmp[3:6] = [0.0, 0.0, 0.0]
		tmp[6:] = -(a[i][0]*b[i])
		A[2*i] = tmp

		tmp[0:3] = [0.0, 0.0, 0.0]
		tmp[3:6] = b[i]
		tmp[6:] = -(a[i][1]*b[i])
		A[2*i+1] = tmp

	AT_A = np.dot(A.T, A)
	eigenValues, eigenVectors = np.linalg.eig(AT_A)
	# print(eigenValues)
	# print(eigenVectors)

	# Find H, which is the eigenvector with the smallest eigenvalue
	idx = eigenValues.argsort()[0]
	# eigenValues = eigenValues[idx]
	# The column of the returned eigenvector is real eigen vector.
	H = eigenVectors[:, idx]
	H = H.reshape((3, 3))
	# print("Test ||H|| (sqrt(sum(H^2))): %3lf" % np.sqrt(np.sum(H**2)))
	return H

def GetRectangle(img, corners):
	"""
	Return: [[x1, y1, 0],[x2, y2, 0],...]
	"""
	rectangle_homo = []

	# Get the outer bound of the rectangle
	min_x = img.shape[1]
	max_x = 0
	min_y = img.shape[0]
	max_y = 0
	for i in corners:
		if i[0] > max_x: 
			max_x = i[0]
		if i[0] < min_x:
			min_x = i[0]
		if i[1] > max_y:
			max_y = i[1]
		if i[1] < min_y:
			min_y = i[1]

	# Calculate pixels in constrained area
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if min_x <= j <= max_x and min_y <= i <= max_y:
				# img[i][j] = [255, 255, 0]

				# Left hand side method:
				# To find pixels within 4 corners
				# AX + BY + C = 0
				# D = A * xp + B * yp + C
				# P3 - P1
				A0 = -(corners[2][1] - corners[0][1])
				B0 = (corners[2][0] - corners[0][0])
				C0 = -((A0 * corners[0][0]) + (B0 * corners[0][1]))
				D0 = A0 * j + B0 * i + C0

				# P4 - P3
				A1 = -(corners[3][1] - corners[2][1])
				B1 = (corners[3][0] - corners[2][0])
				C1 = -((A1 * corners[2][0]) + (B1 * corners[2][1]))
				D1 = A1 * j + B1 * i + C1

				# P2 - P4
				A2 = -(corners[1][1] - corners[3][1])
				B2 = (corners[1][0] - corners[3][0])
				C2 = -((A2 * corners[3][0]) + (B2 * corners[3][1]))
				D2 = A2 * j + B2 * i + C2

				# P1 - P2
				A3 = -(corners[0][1] - corners[1][1])
				B3 = (corners[0][0] - corners[1][0])
				C3 = -((A3 * corners[1][0]) + (B3 * corners[1][1]))
				D3 = A3 * j + B3 * i + C3

				if(D0 <= 0 and D1 <= 0 and D2 <= 0 and D3 <= 0):
					# img[i][j] = [0, 255, 255]
					img[i][j] = [0, 0, 0]
					rectangle_homo += [[j, i, 1]]


	# Print the 4 corners
	# for i in corners:
	# 	x = i[0]
	# 	y = i[1]
	# 	# circle(image, coordinate, radius, color, thickness)
	# 	cv2.circle(img, (x, y), 5, (255, 0, 255), -1)

	# cv2.imshow(testA, img)
	# cv2.waitKey(0)

	return rectangle_homo

def ForwardWarp(img_dst, img_src, src_points, H):
	for p in src_points:
		p = p.reshape((3, 1))
		p_prime = np.dot(H, p)
		p_prime /= p_prime[2]
		x = int(np.round(p_prime[0]))
		y = int(np.round(p_prime[1]))
		# print(p[0], p[1])
		# print(x, y)
		img_dst[y][x] = img_src[p[1][0]][p[0][0]]

	return img_dst

def BackwardWarp(img_dst, img_src, dst_points, H):
	for p_prime in dst_points:
		p_prime = p_prime.reshape((3, 1))
		p = np.dot(np.linalg.inv(H), p_prime)
		p /= p[2]
		x = int(np.round(p[0]))
		y = int(np.round(p[1]))
		# print(p_prime[0], p_prime[1])
		# print(x, y)
		img_dst[p_prime[1][0]][p_prime[0][0]] = img_src[y][x]

	return img_dst

def SwapImgA(img, a, b, H, warp='backward'):
	"""
	a = Hb
	"""
	img_out = img.copy()

	Rec_b = GetRectangle(img_out, b)
	Rec_b = np.asarray(Rec_b)
	Rec_a = GetRectangle(img_out, a)
	Rec_a = np.asarray(Rec_a)

	if warp == 'backward':
		# From b to a
		img_out = BackwardWarp(img_out, img, Rec_a, H)
		# From a to b
		img_out = BackwardWarp(img_out, img, Rec_b, np.linalg.inv(H))

	if warp == 'forward':
		# From b to a
		img_out = ForwardWarp(img_out, img, Rec_b, H)
		# From a to b
		img_out = ForwardWarp(img_out, img, Rec_a, np.linalg.inv(H))

	# cv2.imshow('imgA_' + warp, img_out)
	# cv2.waitKey(0)
	cv2.imwrite('imgA_' + warp + '.jpg', img_out)

def SwapImgBC(img_B, img_C, b, c, H, warp='backward'):
	"""
	b = Hc
	"""
	img_B_out = img_B.copy()
	img_C_out = img_C.copy()

	Rec_c = GetRectangle(img_C_out, c)
	Rec_c = np.asarray(Rec_c)
	Rec_b = GetRectangle(img_B_out, b)
	Rec_b = np.asarray(Rec_b)

	if warp == 'backward':
		# From c to b
		img_B_out = BackwardWarp(img_B_out, img_C, Rec_b, H)
		# From b to c
		img_C_out = BackwardWarp(img_C_out, img_B, Rec_c, np.linalg.inv(H))

	if warp == 'forward':
		# From c to b
		img_B_out = ForwardWarp(img_B_out, img_C, Rec_c, H)
		# From b to c
		img_C_out = ForwardWarp(img_C_out, img_B, Rec_b, np.linalg.inv(H))

	# cv2.imshow('ImgB', img_B_out)
	# cv2.waitKey(0)
	# cv2.imshow('ImgC', img_C_out)
	# cv2.waitKey(0)
	cv2.imwrite('imgB_' + warp + '.jpg', img_B_out)
	cv2.imwrite('imgC_' + warp + '.jpg', img_C_out)

if __name__ == '__main__':

	# Load the images.
	testA = 'testA.jpg'
	testB = 'testB.jpg'
	testC = 'testC.jpg'
	img_A = cv2.imread(testA) # H X W X 3
	img_B = cv2.imread(testB)
	img_C = cv2.imread(testC)

	# Load the 2d points
	testA_left = np.load('testA_2D_left.npy') # X(W), Y(H)
	testA_right = np.load('testA_2D_right.npy')
	testB_2D = np.load('testB_2D.npy')
	testC_2D = np.load('testC_2D.npy')

	# Generate homogeneous matrix
	testA_left_homo = GetHomogeneousPoints(testA_left)
	testA_right_homo = GetHomogeneousPoints(testA_right)
	testB_2D_homo = GetHomogeneousPoints(testB_2D)
	testC_2D_homo = GetHomogeneousPoints(testC_2D)

	H_A_left_A_right = GetHomography(testA_left_homo, testA_right_homo)
	H_B_C = GetHomography(testB_2D_homo, testC_2D_homo)

	SwapImgA(img_A, testA_left, testA_right, H_A_left_A_right, warp='forward')
	SwapImgBC(img_B, img_C, testB_2D, testC_2D, H_B_C, warp='forward')

	SwapImgA(img_A, testA_left, testA_right, H_A_left_A_right, warp='backward')
	SwapImgBC(img_B, img_C, testB_2D, testC_2D, H_B_C, warp='backward')
