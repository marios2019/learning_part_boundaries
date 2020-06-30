import numpy as np

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps

def eulerAnglesMatrix(thetax=0.0, thetay=0.0, thetaz=0.0):
	""" 3D rotation matrix based on euler angles """
	
	return np.matmul(np.matmul(rotMatrixY(thetay), rotMatrixZ(thetaz)), rotMatrixX(thetax))


def rotMatrixX(thetax = 0.0):
	""" 3D rotation matrix around x-axis """
	thetax = np.radians(thetax)
	cosx, sinx = np.cos(thetax), np.sin(thetax)

	return np.array(((1.0, 0.0, 0.0), (0.0, cosx, -sinx), (0.0, sinx, cosx)))


def rotMatrixY(thetay = 0.0):
	""" 3D rotation matrix around y-axis """
	thetay = np.radians(thetay)
	cosy, siny = np.cos(thetay), np.sin(thetay)

	return np.array(((cosy, 0.0, siny), (0.0, 1.0, 0.0), (-siny, 0.0, cosy)))


def rotMatrixZ(thetaz = 0.0):
	""" 3D rotation matrix around z-axis """
	thetaz = np.radians(thetaz)
	cosz, sinz = np.cos(thetaz), np.sin(thetaz)

	return np.array(((cosz, -sinz, 0.0), (sinz, cosz, 0.0), (0.0, 0.0, 1.0)))



def axisAngleMatrix(angle=0.0, axis=np.array([1, 0, 0]), angleType='deg'):
	""" 3D roation matrix based on axis-angle representation """

	assert(axis.shape == (3,))
	assert(angleType == 'deg' or angleType == 'rad')

	# Normalize axis
	axis = axis / np.linalg.norm(axis)

	# Use Rodrigues' rotation formula to form a 3D rotation matrix
	if angleType == 'deg':
		angle = np.radians(angle)
	K = skewSymmetric(axis)
	R = np.eye(3, dtype=np.float32) + np.sin(angle)*K + (1-np.cos(angle))*np.matmul(K, K)

	return R


def axisAngleMatrixBatch(angle, rotation_axis, angleType='deg'):
	""" 3D roation matrix based on axis-angle representation """

	assert(angleType == 'deg' or angleType == 'rad')

	# Normalize axis
	axis_norm = np.linalg.norm(rotation_axis, axis=1, keepdims=True)
	if axis_norm.dtype == np.float32:
		axis_norm[axis_norm < _THRESHOLD_TOL_32] = _THRESHOLD_TOL_32
	elif axis_norm.dtype == np.float64:
		axis_norm[axis_norm < _THRESHOLD_TOL_64] = _THRESHOLD_TOL_64
	rotation_axis = rotation_axis / axis_norm

	# Use Rodrigues' rotation formula to form a 3D rotation matrix
	if angleType == 'deg':
		angle = np.radians(angle)
	K = skewSymmetricBatch(rotation_axis)
	eye = np.eye(3, dtype=np.float32)
	R = np.tile(eye, (rotation_axis.shape[0], 1, 1)) + np.sin(angle[:, np.newaxis, np.newaxis])*K \
		+ (1-np.cos(angle[:, np.newaxis, np.newaxis]))*np.linalg.matrix_power(K, 2)

	return R


def skewSymmetric(x=np.array([1, 1, 1])):
	""" Return a skew symmetric matrix (A == -A.T) of a 3d vector """

	assert(x.shape==(3,))

	return np.array([[0.0, -x[2], x[1]],
					 [x[2], 0.0, -x[0]],
					 [-x[1], x[0], 0.0]], dtype=np.float32)


def skewSymmetricBatch(x):
	""" Return a skew symmetric matrix (A == -A.T) of a 3d vector """

	K = np.zeros((x.shape[0], 3, 3), dtype=np.float32)
	K[:,0,1] = -x[:,2]; K[:,0,2] = x[:,1]; K[:,1,0] = x[:,2]; K[:,1,2] = -x[:,0]; K[:,2,0] = -x[:,1]; K[:,2,1] = x[:,0]

	return K
