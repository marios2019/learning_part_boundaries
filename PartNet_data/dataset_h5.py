""" Import PartNet semantic and boundary segmentation dataset """

import os
import numpy as np
import h5py
import sys
from collections import OrderedDict
from progressbar import ProgressBar


BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
from utils import rotations3D

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps


# PartNet categories
categories = ["Bag", "Bed", "Bottle", "Bowl", "Chair", "Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
              "Hat", "Keyboard", "Knife", "Lamp", "Laptop", "Microwave", "Mug", "Refrigerator", "Scissors",
              "StorageFurniture", "Table", "TrashCan", "Vase"]
categories_dict = dict(zip(categories, range(len(categories))))


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
	nNonBoundaries = np.sum(labels == 0)
	nBoundaries = np.sum(labels == 1)
	assert (nNonBoundaries + nBoundaries == labels.shape)
	weights = np.ones(labels.shape)
	weights[labels == 1] = float(nNonBoundaries) / float(np.maximum(nBoundaries, 1))

	return np.expand_dims(weights.astype(np.float32), axis=1)



class Dataset():
	def __init__(self, pc_dataset_dir='', split='', sampling=True, npoints=10000, normalize=True, nrotations=1,
				 random_rotation=False, rotate_principals_around_normal=False, category='', return_cat=False, return_weights=False,
				 evaluate=False):

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
		self.return_cat = return_cat
		self.return_weights = return_weights
		self.category = category

		# Check if given category exists and specify level -> only fine-grained semantic levels of PartNet are used
		cat_directories = os.listdir(self.pc_dataset_dir)
		cat_found = False
		for cat_fn in cat_directories:
			if cat_fn.split('-')[0].lower() == self.category.lower():
				CATEGORY_DIR = os.path.join(self.pc_dataset_dir, cat_fn)
				self.level = int(cat_fn.split('-')[-1])
				cat_found = True
		if not cat_found:
			print("ERROR: {cat:s} is not part of PartNet dataset".format(cat=self.category))
			exit(-1)
		self.CATEGORY_DIR = CATEGORY_DIR

		# Split check
		validSplits = ["train", "test", "val"]
		if not self.split in validSplits:
			print("Error: unknown {split:s}; valid splits: train, test, val".format(split=self.split))
			exit(-1)

		# Read filelist
		splitFileListFN = os.path.join(self.CATEGORY_DIR, "{split:s}_files.txt".format(split=self.split))
		if not os.path.isfile(splitFileListFN):
			print("ERROR: {fn:s} does not exists" .format(fn=splitFileListFN))
			exit(-1)
		self.splitFileListFN = splitFileListFN
		with open(self.splitFileListFN, "rb") as f:
			self.splitFilelist = [os.path.join(CATEGORY_DIR, os.path.basename(line.strip())) for line in f.readlines()]
		self.splitFilelist.sort(key=lambda x: int(os.path.basename(x).split("-")[-1].split('.')[0]))

		# Read .h5 files
		print("Loading {datasetFN:s} ..." .format(datasetFN=self.pc_dataset_dir))
		self.points = []; self.seg = []; self.normals = []; self.principal = []
		bar = ProgressBar()
		for idx in bar(range(len(self.splitFilelist))):
			if not os.path.isfile(self.splitFilelist[idx]):
				print("ERROR: {fn:s} does not exists".format(fn=self.splitFilelist[idx]))
				exit(-1)
			segData = h5py.File(self.splitFilelist[idx], 'r')
			points = segData['data'][...].astype(np.float32)
			seg = segData['label_seg'][...].astype(np.float32)
			self.points.append(points)
			self.seg.append(seg)
			self._npoints = len(points[0])
			self.normals.append(segData['normals'][...].astype(np.float32))
			self._normals = True
			self.principal.append(segData['principal'][...].astype(np.float32))
			self._principal = True
			if segData.attrs.keys():
				if 'fn' in dir(self):
					for idx in range(len(points)):
						self.fn.append(segData.attrs[str(idx)])
				else:
					self.fn = []
					for idx in range(len(points)):
						self.fn.append(segData.attrs[str(idx)])

		self.points = np.vstack(self.points)
		self.seg = np.vstack(self.seg)
		self.normals = np.vstack(self.normals)
		self.principal = np.vstack(self.principal)
		self.segClasses = OrderedDict({self.category : range(np.amin(self.seg).astype(np.int32), np.amax(self.seg).astype(np.int32)+1)})

		# Augment data for rotation
		if nrotations < 1:
			print("ERROR: number of rotations should be a positive number")
			exit(-1)
		self.nrotations = nrotations
		self.random_rotation = random_rotation
		self.rotate_principal_around_normal = rotate_principals_around_normal
		self.ptsIdx = np.array([[idx] * nrotations for idx, _ in enumerate(self.points)])
		self.ptsIdx = self.ptsIdx.reshape((len(self.points) * self.nrotations))
		assert (self.ptsIdx.shape[0] / self.nrotations == len(self.points))
		if self.random_rotation:
			self.rotateMap = None
		else:
			self.rotateMap = np.arange(0.0, 360.0, 360.0 / float(self.nrotations), dtype=np.float32)
		self.rotate = 0

		# Print dataset configuration
		print("PartNet Dataset Configuration")
		print("##############################")
		print("Dataset: {dataset:s}" .format(dataset=self.pc_dataset_dir))
		print("Category: {category:s}" .format(category=self.category))
		print("Hierarchy level: {level:d}".format(level=self.level))
		print("Split: {split:s}" .format(split=self.split))
		print("#pts: {nPts:d}". format(nPts=len(self.points)))
		print("#rotations: {nRot:d}".format(nRot=self.nrotations))
		print("Random rotation: {random_rotation:s}".format(random_rotation=str(self.random_rotation)))
		print("Rotate principal around normal: {rotate_principal_around_normal:s}"
			  .format(rotate_principal_around_normal=str(self.rotate_principal_around_normal)))
		print("Subsample: {subsample:s}".format(subsample=str(self.sampling)))
		print("Npoints: {npoints:d}".format(npoints=self.npoints if self.sampling else self._npoints))
		print("Normalize: {normalize:s}".format(normalize=str(self.normalize)))
		print("Return category: {retCat:s}".format(retCat=str(self.return_cat)))
		print("Return weights: {retWeights:s}".format(retWeights=str(self.return_weights)))

		self.evaluate = evaluate
		if evaluate:
			# For random rotations
			self.rng1 = np.random.RandomState(categories_dict[category])
			self.rng2 = np.random.RandomState(categories_dict[category]+len(categories_dict))


	def __getitem__(self, idx):

		pts = self.points[self.ptsIdx[idx]]
		seg = self.seg[self.ptsIdx[idx]]
		normals = self.normals[self.ptsIdx[idx]]
		principal = np.copy(self.principal[self.ptsIdx[idx]])

		if self.normalize:
			# Normalize XYZ cartesian coordinates
			pts = pc_normalize(pts)
			# Check for non-finite values
			assert (np.isfinite(pts).all())

		if self.rotate >= self.nrotations:
			self.rotate = 0
		if self.random_rotation:
			# Rotate 3d points around a random axis
			if self.evaluate:
				axis = self.rng2.uniform(-1, 1, 3)
				angle = self.rng2.rand() * 360.0
			else:
				axis = np.random.uniform(-1, 1, 3)
				angle = np.random.rand() * 360.0
			R = rotations3D.axisAngleMatrix(angle=angle, axis=axis)
		else:
			# Rotate 3d points along y-axis
			R = rotations3D.eulerAnglesMatrix(thetay=self.rotateMap[self.rotate])
		# If #rotations is 1 and random rotation=False then R = I (identity matrix)
		self.rotate += 1
		pts = np.matmul(R, pts.T).T
		if self._normals:
			normals = np.matmul(R, normals.T).T
		if self._principal:
			principal[:, 0:3] = np.matmul(R, principal[:, 0:3].T).T
			principal[:, 3:6] = np.matmul(R, principal[:, 3:6].T).T

		# Random rotation of principal directions around normal -> random tangent vectors
		if self.rotate_principal_around_normal:
			if self.evaluate:
				angles = self.rng1.rand(normals.shape[0]) * 360.0
			else:
				angles = np.random.rand(normals.shape[0]) * 360.0
			R = rotations3D.axisAngleMatrixBatch(angle=angles, rotation_axis=normals)
			principal[:, 0:3] = np.squeeze(np.matmul(R, principal[:, 0:3, np.newaxis]))
			principal[:, 3:6] = np.squeeze(np.matmul(R, principal[:, 3:6, np.newaxis]))

		if self.sampling:
			# Check if there are enough points to sample
			if pts.shape[0] >= self.npoints:
				choice = np.random.choice(pts.shape[0], self.npoints, replace=False)
			else:
				# Sample with replacement
				choice = np.random.choice(pts.shape[0], self.npoints, replace=True)
			# Sample points
			pts = pts[choice]
			seg = seg[choice]
			normals = normals[choice]
			principal = principal[choice]

		pointcloud = (pts,)
		pointcloud += (normals,)
		pointcloud += (principal,)
		pointcloud += (seg,)
		if self.return_weights: # Add weights
			# Calculate points weigths w.r.t the ratio of non-boundary over boundary points
			pointcloud += (point_weights(seg),)
		if self.return_cat:
			pointcloud += (self.category,)

		return pointcloud


	def __len__(self):
		return len(self.ptsIdx)


	def filename(self, idx):
		return self.fn[self.ptsIdx[idx]]
