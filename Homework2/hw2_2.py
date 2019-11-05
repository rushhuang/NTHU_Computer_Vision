import cv2
import numpy as np

def GetHomogeneousPoints(point_2d):
	tmp = np.ones((point_2d.shape[0], 1))
	homo = np.concatenate((point_2d, tmp), axis = 1)
	return homo

def GetHomography(a, b):
	# a = Hb

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
	eigenValues = eigenValues[idx]
	# The column of the returned eigenvector is real eigen vector.
	H = eigenVectors[:, idx]
	H = H.reshape((3, 3))
	return H

def ForwardWarp():
	pass

def BackwardWarp():
	pass

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
	# H_B_C = GetHomography(testB_2D_homo, testC_2D_homo)

	min_x = img_A.shape[1]
	max_x = 0
	min_y = img_A.shape[0]
	max_y = 0
	for i in testA_left:
		if i[0] > max_x: 
			max_x = i[0]
		if i[0] < min_x:
			min_x = i[0]
		if i[1] > max_y:
			max_y = i[1]
		if i[1] < min_y:
			min_y = i[1]

	for i in range(img_A.shape[0]):
		for j in range(img_A.shape[1]):
			if min_x <= j <= max_x and min_y <= i <= max_y:
				img_A[i][j] = [0, 255, 255]

	for i in testA_left:
		x = i[0]
		y = i[1]
		# circle(image, coordinate, radius, color, thickness)
		cv2.circle(img_A, (x, y), 5, (255, 0, 255), -1)

	cv2.imshow(testA, img_A)
	cv2.waitKey(0)

