import cv2
import numpy as np
from visualize import visualize

def getProjectionMatrix(corner_label, threeD_point):
	"""
	Get the projection matrix from given 2D and 3D points.
	
	Parameters:
	* corner_label: 2D points in the image.
	* threeD_points: 3D points from the reality.
	"""

	# Concatenate the equations
	# Ai = [[X1, Y1, Z1, 1, 0, 0, 0, 0, −x1X1, −x1Y1, −x1Z1, −x1],
	#       [0, 0, 0, 0, X1, Y1, Z1, 1, −y1X1, −y1Y1, −y1Z1, −y1],...]
	# A is a (2N, 12) matrix, N is the point number
	A = np.zeros((2*threeD_point.shape[0], 12))
	tmp = [12]
	for i in range(threeD_point.shape[0]):
		tmp[0:4] = threeD_point[i]
		tmp[4:8] = [0.0, 0.0, 0.0, 0.0]
		tmp[8:] = -(corner_label[i][0]*threeD_point[i])
		A[2*i] = tmp

		tmp[0:4] = [0.0, 0.0, 0.0, 0.0]
		tmp[4:8] = threeD_point[i]
		tmp[8:] = -(corner_label[i][1]*threeD_point[i])
		A[2*i+1] = tmp

	AT_A = np.dot(A.T, A)
	eigenValues, eigenVectors = np.linalg.eig(AT_A)
	# print(eigenValues)
	# print(eigenVectors)

	# Find P, which is the eigenvector with the smallest eigenvalue
	idx = eigenValues.argsort()[0]
	eigenValues = eigenValues[idx]
	# The column of the returned eigenvector is real eigen vector.
	P = eigenVectors[:, idx]
	P = P.reshape((3, 4))
	# print(P)
	# print(P.shape)
	# print("Test ||P|| (sqrt(sum(P^2))): %3lf" % np.sqrt(np.sum(P**2)))
	return P

def getKRT(P):
	"""
	Get the intrinsic parameters K, extrinsic parameters R,
	and translation vector t.

	Parameters:
	* P: Projection Matrix
	"""

	# Normalize P with ||P31,P32,P33||
	P = P/np.sqrt(P[2][0]**2 + P[2][1]**2 + P[2][2]**2)

	# Do RQ decomposition to get intrinsic parameter K,
	# and extrinsic parameters R by QR decomposition
	M = P[:3, :3]
	Q, R = np.linalg.qr(np.linalg.inv(M))
	K = np.linalg.inv(R)
	R = np.linalg.inv(Q)

	# Compute t, translation vector
	P_T = P[:3, 3].reshape(3, 1)
	t = np.dot(np.linalg.inv(K), P_T)

	# print("K: ")
	# print(K)
	# print("R: ")
	# print(R)
	# print(np.dot(R, R.T))
	# print("t: ")
	# print(t)

	return K, R, t

def ReProject(img, img_name, K, R, t, twoD, threeD):
	"""
	Reproject the 2d points on image by given K, R, and t.

	Parameters:
	* img: Image to reproject on.
	* img_name: Filename of the image to save.
	* K: 3X3 intrinsic parameters.
	* R: 3X3 extrinsic parameters rotation matrix.
	* t: 3X1 extrinsic parameters translation vector.
	* twoD: 2D points that user labeled.
	* threeD: 3D points of the 2D points in real world.
	"""

	# Concatenate the extrinsic matrix
	extrinsic = R
	extrinsic = np.append(extrinsic, t, axis=1)
	a = np.array([0.0, 0.0, 0.0, 1.0]).reshape((1, 4))
	extrinsic = np.append(extrinsic, a, axis=0)

	projection = np.array([ [1, 0, 0, 0],
							[0, 1, 0, 0],
							[0, 0, 1, 0]])

	P_new = np.dot(np.dot(K, projection), extrinsic)
	# print(P_new)

	for i in twoD:
		x = i[0]
		y = i[1]
		# circle(image, coordinate, radius, color, thickness)
		cv2.circle(img, (x, y), 5, (0, 255, 255), -1)

	tgs = []
	for j in threeD:
		tg = np.dot(P_new, j)
		tg = tg/tg[2]
		x = int(round(tg[0]))
		y = int(round(tg[1]))
		cv2.circle(img, (x, y), 5, (255, 0, 255), 2)
		tgs += [[x, y]]

	tgs = np.asarray(tgs)
	rmse = np.sqrt(np.mean((tgs - twoD)**2))
	# print("Root Mean Square Error(RMSE): %f" % rmse)

	cv2.imwrite(img_name, img)
	# cv2.imshow(img_name, img)
	# cv2.waitKey(0)


if __name__ == '__main__':

	# Open the image
	img_name_1 = 'data/chessboard_1.jpg'
	img_name_2 = 'data/chessboard_2.jpg'
	img_1 = cv2.imread(img_name_1)
	img_2 = cv2.imread(img_name_2)
	# imageshow(img, savefile=False)

	# Open the corner label file
	corner_label_1 = np.load('Point2D_1.npy')
	corner_label_2 = np.load('Point2D_2.npy')
	
	# Open the 3D points file
	threeD_point = np.loadtxt('data/Point3D.txt')
	
	# Make homogeneous coordinates of corner label and 3D points
	tmp = np.ones((corner_label_1.shape[0], 1))
	corner_label_1_homo = np.concatenate((corner_label_1, tmp), axis = 1)
	tmp = np.ones((corner_label_2.shape[0], 1))
	corner_label_2_homo = np.concatenate((corner_label_2, tmp), axis = 1)
	tmp = np.ones((threeD_point.shape[0], 1))
	threeD_point_homo = np.concatenate((threeD_point, tmp), axis = 1)

	### Calculate projection matrix, P ###
	P_1 = getProjectionMatrix(corner_label_1_homo, threeD_point_homo)
	P_2 = getProjectionMatrix(corner_label_2_homo, threeD_point_homo)

	### Calculate parameters K, R, and t.###
	K_1, R_1, t_1 = getKRT(P_1)
	K_2, R_2, t_2 = getKRT(P_2)

	### Reproject 2D points by K, R, T ###
	ReProject(img_1, 'chessboard1_r.jpg', K_1, R_1, t_1,
				corner_label_1, threeD_point_homo)
	ReProject(img_2, 'chessboard2_r.jpg', K_2, R_2, t_2,
				corner_label_2, threeD_point_homo)

	visualize(threeD_point, R_1, t_1, R_2, t_2)