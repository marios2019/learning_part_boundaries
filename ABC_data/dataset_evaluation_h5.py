""" Import ABC boundary evaluation dataset """

import os
import numpy as np
import h5py
import sys
from collections import OrderedDict
from progressbar import ProgressBar

# Import custom packages
BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
from utils import rotations3D

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps


def pc_normalize(pc):
	"""Center points"""
	centroid = np.mean(pc, axis=0)
	pc = pc - centroid
	# Normalize points
	m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
	pc /= np.maximum(m, _THRESHOLD_TOL_64 if m.dtype==np.float64 else _THRESHOLD_TOL_32)

	return pc


def point_weights(labels):
	""" Non-Boundary point weight = 1
	Boundary point weight = #non_boundary_points/#boundary_points"""
	n_non_boundaries = np.sum(labels == 0)
	n_boundaries = np.sum(labels == 1)
	assert (n_non_boundaries + n_boundaries == labels.shape)
	weights = np.ones(labels.shape)
	weights[labels == 1] = float(n_non_boundaries) / float(np.maximum(n_boundaries, 1))

	return np.expand_dims(weights.astype(np.float32), axis=1)


class Dataset():
	def __init__(self, pc_dataset_dir='', split='', sampling=True, npoints=1000, normalize=True, nrotations=1,
				 random_rotation=False, rotate_principals_around_normal=False):

		# Check dataset directory
		if not os.path.isdir(pc_dataset_dir):
			print("'{dataset:s}' missing directory".format(dataset=pc_dataset_dir))
			exit(-1)

		# Point cloud configuration
		self.pc_dataset_dir = pc_dataset_dir
		self.split = split.lower()
		self.npoints = npoints
		self.normalize = normalize
		self.sampling = sampling # if sampling=True and npoints=10000 -> randomize point cloud data

		# Split check
		valid_splits = ["test", "val"]
		if not self.split in valid_splits:
			print("Error: unknown {split:s}; valid splits: train, test, val".format(split=self.split))
			exit(-1)

		# Read filelist
		split_file_list_fn = os.path.join(self.pc_dataset_dir, "{split:s}_files.txt".format(split=self.split))
		if not os.path.isfile(split_file_list_fn):
			print("ERROR: {fn:s} does not exists" .format(fn=split_file_list_fn))
			exit(-1)
		self.split_file_list_fn = split_file_list_fn
		with open(self.split_file_list_fn, "rb") as f:
			self.split_file_list = [os.path.join(self.pc_dataset_dir, os.path.basename(line.strip())) for line in f.readlines()]
		self.split_file_list.sort(key=lambda x: int(os.path.basename(x).split("-")[-1].split('.')[0]))

		# Read .h5 files
		print("Loading {dataset_fn:s} ..." .format(dataset_fn=self.pc_dataset_dir))
		self.points = []; self.seg = []; self.normals = []; self.principal = []
		bar = ProgressBar()
		for idx in bar(range(len(self.split_file_list))):
			if not os.path.isfile(self.split_file_list[idx]):
				print("ERROR: {fn:s} does not exists".format(fn=self.split_file_list[idx]))
				exit(-1)
			seg_data = h5py.File(self.split_file_list[idx], 'r')
			# Load xyz coordinates and boundary labels
			points = seg_data['data'][...].astype(np.float32)
			seg = seg_data['label_seg'][...].astype(np.float32)
			self.points.append(points)
			self.seg.append(seg)
			self._npoints = len(points[0])
			# Load normals
			self.normals.append(seg_data['normals'][...].astype(np.float32))
			self._normals = True
			# Load principal directions -> these will be rotated along normal to get random tangent vectors
			self.principal.append(seg_data['principal'][...].astype(np.float32))
			if seg_data.attrs.keys():
				# Load model ids
				if 'fn' in dir(self):
					for fn_idx in range(len(points)):
						self.fn.append(seg_data.attrs[str(fn_idx)])
				else:
					self.fn = []
					for fn_idx in range(len(points)):
						self.fn.append(seg_data.attrs[str(fn_idx)])

		# Cast lists to np.array() objects
		self.points = np.vstack(self.points).astype(np.float32)
		self.seg = np.vstack(self.seg).astype(np.float32)
		self.normals = np.vstack(self.normals).astype(np.float32)
		self.principal = np.vstack(self.principal).astype(np.float32)

		# Augment data for rotation
		if nrotations < 1:
			print("ERROR: number of rotations should be a positive number")
			exit(-1)
		self.nrotations = nrotations
		self.random_rotation = random_rotation
		self.rotate_principal_around_normal = rotate_principals_around_normal
		self.pts_idx = np.array([[idx] * nrotations for idx, _ in enumerate(self.points)])
		self.pts_idx = self.pts_idx.reshape((len(self.points) * self.nrotations))
		assert (self.pts_idx.shape[0] / self.nrotations == len(self.points))
		if self.random_rotation:
			self.rotate_map = None
		else:
			self.rotate_map = np.arange(0.0, 360.0, 360.0 / float(self.nrotations), dtype=np.float32)
		self.rotate = 0

		if self.normalize:
			# Normalize XYZ cartesian coordinates
			print("Normalize point clouds ...")
			for p_idx, p in enumerate(self.points):
				p = pc_normalize(p)
				# Check for non-finite values
				assert (np.isfinite(p).all())
				self.points[p_idx] = p
			print("DONE")

		# Print dataset configuration
		print("ABC Dataset Configuration")
		print("##############################")
		print("Dataset: {dataset:s}" .format(dataset=self.pc_dataset_dir))
		print("Split: {split:s}" .format(split=self.split))
		print("#pts: {nPts:d}". format(nPts=len(self.points)))
		print("#rotations: {nRot:d}".format(nRot=self.nrotations))
		print("Random rotation: {random_rotation:s}".format(random_rotation=str(self.random_rotation)))
		print("Rotate principal around normal: {rotate_principal_around_normal:s}"
			  .format(rotate_principal_around_normal=str(self.rotate_principal_around_normal)))
		print("Subsample: {subsample:s}".format(subsample=str(self.sampling)))
		print("Npoints: {npoints:d}".format(npoints=self.npoints if self.sampling else -1))
		print("Normalize: {normalize:s}".format(normalize=str(self.normalize)))

		# For random rotations
		self.rng1 = np.random.RandomState(0)
		self.rng2 = np.random.RandomState(1)


	def __getitem__(self, idx):

		# Get xyz coordinates and boundary labels for the i-th point cloud
		pts = self.points[self.pts_idx[idx]]
		seg = self.seg[self.pts_idx[idx]]
		# Get normals
		normals = self.normals[self.pts_idx[idx]]
		# Get principal directions
		principal = np.copy(self.principal[self.pts_idx[idx]])

		if self.rotate >= self.nrotations:
			self.rotate = 0
		if self.random_rotation:
			# Rotate 3d points around a random axis
			axis = self.rng1.uniform(-1, 1, 3)
			angle = self.rng1.rand() * 360.0
			R = rotations3D.axisAngleMatrix(angle=angle, axis=axis)
		else:
			# Rotate 3d points along y-axis
			R = rotations3D.eulerAnglesMatrix(thetay=self.rotate_map[self.rotate])
		# If #rotations is 1 and random rotation=False then R = I (identity matrix)
		R = R.astype(np.float32)
		self.rotate += 1
		pts = np.matmul(R, pts.T).T
		normals = np.matmul(R, normals.T).T
		principal[:, 0:3] = np.matmul(R, principal[:, 0:3].T).T
		principal[:, 3:6] = np.matmul(R, principal[:, 3:6].T).T

		# Random rotation of principal directions around normal -> random tangent vectors
		if self.rotate_principal_around_normal:
			# Principals only for non boundary (surface) points
			ind = np.where(seg==0)[0]
			angles = self.rng2.rand(normals[ind].shape[0]) * 360.0
			R = rotations3D.axisAngleMatrixBatch(angle=angles, rotation_axis=normals[ind])
			principal[:, 0:3] = np.squeeze(np.matmul(R, principal[:, 0:3, np.newaxis]))
			principal[:, 3:6] = np.squeeze(np.matmul(R, principal[:, 3:6, np.newaxis]))

		if self.sampling:
			# Sample only non boundary (surface) points
			ind = np.where(seg == 0)[0]
			non_boundary_pts = pts[ind]
			# Check if there are enough points to sample
			if non_boundary_pts.shape[0] >= self.npoints:
				choice = np.random.choice(non_boundary_pts.shape[0], self.npoints, replace=False)
			else:
				# Sample with replacement
				choice = np.random.choice(non_boundary_pts.shape[0], self.npoints, replace=True)
			# Sample points
			non_boundary_pts = non_boundary_pts[choice]
			non_boundary_normals = normals[ind]
			non_boundary_normals = non_boundary_normals[choice]
			principal = principal[choice]

			# Concatenate non boundary and boundary points and normals
			ind = np.where(seg==1)[0]
			boundary_pts = pts[ind]
			boundary_normals = normals[ind]
			pts = np.vstack((non_boundary_pts, boundary_pts))
			normals = np.vstack((non_boundary_normals, boundary_normals))
			seg = np.zeros((pts.shape[0],), dtype=np.float32)
			seg[non_boundary_pts.shape[0]:] = 1

		pointcloud = (pts,)
		pointcloud += (normals,)
		pointcloud += (principal,)
		pointcloud += (seg,)

		return pointcloud


	def __len__(self):
		return len(self.pts_idx)


	def filename(self, idx):
		return self.fn[self.pts_idx[idx]]