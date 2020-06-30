"""
Utilities used for evaluation
"""
import numpy as np
from scipy import spatial
import rotations3D

_THRESHOLD_TOL_32 = 2.0 * np.finfo(np.float32).eps
_THRESHOLD_TOL_64 = 2.0 * np.finfo(np.float32).eps


def get_pointcloud(dataset, idx, n_features, get_tangent_vectors=False):
	"""Get i-th pointcloud from specified dataset"""

	# Get data
	points, normals, tangent_vectors, seg = dataset[idx]
	pointcloud_features = np.hstack((normals, tangent_vectors, seg[:, np.newaxis]))

	# Accumulate input features
	input_features = np.zeros((points.shape[0], n_features))
	input_features[:, 0:3] = points
	if get_tangent_vectors:
		input_features[:, 3:9] = tangent_vectors

	return points, input_features, pointcloud_features


def get_pointcloud_evaluation(dataset, idx, n_features, get_tangent_vectors=True):
	"""Get i-th pointcloud from specified dataset"""

	# Get data
	points, normals, tangent_vectors, seg = dataset[idx]
	ind = np.where(seg==0)[0]
	non_boundary_points = points[ind]
	non_boundary_normals = normals[ind]
	ind = np.where(seg==1)[0]
	boundary_points = points[ind]
	boundary_normals = normals[ind]
	pointcloud_features = np.hstack((non_boundary_normals, tangent_vectors, seg[seg==0, np.newaxis]))

	# Accumulate input features
	input_features = np.zeros((non_boundary_points.shape[0], n_features))
	input_features[:, 0:3] = non_boundary_points
	if get_tangent_vectors:
		input_features[:, 3:9] = tangent_vectors

	return non_boundary_points, input_features, pointcloud_features, boundary_points, boundary_normals


def mIoU_calculation(seg, segp, shape_iou_tot, shape_iou_cnt, part_intersect, part_union, num_semantic_classes):
	"""Calculate shape and part category mIoU"""
	seg_res = segp
	seg_res[seg == 0] = 0
	cur_pred = seg_res
	cur_gt = seg
	cur_shape_iou_tot = 0.0
	cur_shape_iou_cnt = 0
	for j in range(1, num_semantic_classes):
		cur_gt_mask = (cur_gt == j)
		cur_pred_mask = (cur_pred == j)

		has_gt = (np.sum(cur_gt_mask) > 0)
		has_pred = (np.sum(cur_pred_mask) > 0)

		if has_gt or has_pred:
			intersect = np.sum(cur_gt_mask & cur_pred_mask)
			union = np.sum(cur_gt_mask | cur_pred_mask)
			iou = intersect / float(union)

			cur_shape_iou_tot += iou
			cur_shape_iou_cnt += 1

			part_intersect[j] += intersect
			part_union[j] += union

	if cur_shape_iou_cnt > 0:
		cur_shape_miou = cur_shape_iou_tot / float(cur_shape_iou_cnt)
		shape_iou_tot += cur_shape_miou
		shape_iou_cnt += 1

	return (shape_iou_tot, shape_iou_cnt, part_intersect, part_union)


def precision(boundary_points_gt, boundary_points_pred, dist):
	""" Defined as the fraction of predicted boundary points in a point cloud that are near any annotated boundary """

	if len(boundary_points_pred) == 0:
		return 0.0, 0, 0

	boundary_points_gt_KDTree = spatial.cKDTree(boundary_points_gt, copy_data=False, balanced_tree=False, compact_nodes=False)
	nn_idx = boundary_points_gt_KDTree.query_ball_point(boundary_points_pred, dist)

	TP = 0; FP = 0
	for nn in nn_idx:
		if len(nn):
			TP += 1
		else:
			FP += 1
	assert((TP + FP) == len(boundary_points_pred))

	try:
		P = TP / float(TP+FP)
	except ZeroDivisionError:
		P = 0.0

	return P, TP, FP


def recall(boundary_points_gt, boundary_points_pred, dist):
	""" Defined as the fraction of annotated boundary points that are near any predicted boundary point """

	if len(boundary_points_pred) == 0:
		return 0.0, 0, len(boundary_points_gt)

	boundary_points_pred_KDTree = spatial.cKDTree(boundary_points_pred, copy_data=False, balanced_tree=False, compact_nodes=False)
	nn_idx = boundary_points_pred_KDTree.query_ball_point(boundary_points_gt, dist)

	TP = 0; FN = 0
	for nn in nn_idx:
		if len(nn):
			TP += 1
		else:
			FN += 1
	assert((TP + FN) == len(boundary_points_gt))

	try:
		R = TP / float(TP+FN)
	except ZeroDivisionError:
		R = 0.0

	return R, TP, FN


def chamfer_distance(**kwargs):

	if len(kwargs.keys()) == 3:
		if "segmentation1" in kwargs.keys():
			mandatory_args = ["segmentation2", "points"]
			for arg in mandatory_args:
				if arg not in kwargs.keys():
					print("ERROR: Argument {arg:s} is missing" .format(arg=arg))
					exit(-1)
			return _chamfer_distance_1(segmentation1=kwargs["segmentation1"], segmentation2=kwargs["segmentation2"],
									  points=kwargs["points"])
		elif "segmentation1_points" in kwargs.keys():
			mandatory_args = ["segmentation2_points", "points"]
			for arg in mandatory_args:
				if arg not in kwargs.keys():
					print("ERROR: Argument {arg:s} is missing".format(arg=arg))
					exit(-1)
			return _chamfer_distance_2(segmentation1_points=kwargs["segmentation1_points"],
									  segmentation2_points=kwargs["segmentation2_points"],
									  points=kwargs["points"])
		else:
			print("ERROR: Invalid arguments")
			exit(-1)
	else:
		print("ERROR: Invalid nubmer of arguments")
		exit(-1)


def _chamfer_distance_1(segmentation1, segmentation2, points):
	"""Calculate chamfer distance between segmentations 1 and 2 and vice versa"""

	directional_mean_dist = []
	for i in range(2):
		# Calculate CD for both directions (s1 -> s2, s2 -> s1)
		if i == 0:
			s1 = segmentation1; s2 = segmentation2
		else:
			s1 = segmentation2; s2 = segmentation1

		# Find the boundary points of s1 and s2 segmentations
		boundary_points_s1 = points[s1 == 1]
		boundary_points_s2 = points[s2 == 1]

		# Create kd-tree for boundary points in s2
		s2_KD_Tree = spatial.cKDTree(boundary_points_s2, copy_data=False, balanced_tree=False, compact_nodes=False)

		# Find s2 boundary points 1-nn for each point in s1
		nn_dist, _ = s2_KD_Tree.query(boundary_points_s1, k=1)
		directional_mean_dist.append(np.mean(nn_dist))

	DCD = (directional_mean_dist[0] + directional_mean_dist[1]) / pc_bounding_box_diagonal(points)

	return DCD


def _chamfer_distance_2(segmentation1_points, segmentation2_points, points):
	"""Calculate chamfer distance between segmentations 1 and 2 and vice versa"""

	directional_mean_dist = []
	for i in range(2):
		# Calculate CD for both directions (s1 -> s2, s2 -> s1)
		if i == 0:
			s1 = segmentation1_points; s2 = segmentation2_points
		else:
			s1 = segmentation2_points; s2 = segmentation1_points

		# Find the boundary points of s1 and s2 segmentations
		boundary_points_s1 = s1
		boundary_points_s2 = s2

		# Create kd-tree for boundary points in s2
		s2_KD_Tree = spatial.cKDTree(boundary_points_s2, copy_data=False, balanced_tree=False, compact_nodes=False)

		# Find s2 boundary points 1-nn for each point in s1
		nn_dist, _ = s2_KD_Tree.query(boundary_points_s1, k=1)
		directional_mean_dist.append(np.mean(nn_dist))

	DCD = (directional_mean_dist[0] + directional_mean_dist[1]) / pc_bounding_box_diagonal(points)

	return DCD


def pc_bounding_box_diagonal(pc):
	"""Calculate point cloud's bounding box"""
	centroid = np.mean(pc, axis=0)
	pc2 = pc - centroid

	# Calculate bounding box diagonal
	xyz_min = np.amin(pc2, axis=0)
	xyz_max = np.amax(pc2, axis=0)
	bb_diagonal = np.max([np.linalg.norm(xyz_max - xyz_min, ord=2), _THRESHOLD_TOL_64 if pc.dtype==np.float64 else _THRESHOLD_TOL_32])

	return bb_diagonal


def majority(labels):
	"""Find majority label"""
	idx, ctr = 0, 1
	for i in range(1, len(labels)):
		if labels[i] == labels[idx]:
			ctr += 1
		else:
			ctr -= 1
			if ctr == 0:
				idx = i
				ctr = 1

	return labels[idx], idx, ctr


def boundary_annotation(points, semantic_seg):
	"""Boundary annotate points w.r.t. semantic segmentation"""

	assert(len(points) == len(semantic_seg))

	# Find maximum sampling distance
	points_KD_tree = spatial.cKDTree(points, copy_data=False, balanced_tree=False, compact_nodes=False)
	# Find 1-nn distance for all points
	nn_dist, _ = points_KD_tree.query(points, k=2)
	# Find maximum sampling distance ~ Poisson disk radius
	maximum_sampling_distance = np.amax(nn_dist[:, 1])

	# Annotate a point as a boundary if 1 of its neighbors within a ball radius=maximum_sampling_distance
	# has a different semantic label
	boundaries = np.zeros_like(semantic_seg)
	nn_idx = points_KD_tree.query_ball_point(points, maximum_sampling_distance)
	for ind, nn in enumerate(nn_idx):
		_, _, ctr = majority(semantic_seg[nn])
		if ctr < len(nn):
			boundaries[ind] = 1.0

	return boundaries


### Noise utilities


def jitter_points(points, sigma=0.005, clip=0.01, rng=None, return_noise=False):
	""" Jitter points """
	assert((points.ndim == 2) or (points.ndim == 3))

	if points.ndim == 2:
		points = np.expand_dims(points, axis=0)

	# Get random noise
	if rng is not None:
		noise = np.clip(sigma * rng.randn(points.shape[0], points.shape[1], points.shape[2]), -clip, clip)
	else:
		noise = np.clip(sigma * np.random.randn(points.shape[0], points.shape[1], points.shape[2]), -clip, clip)

	if return_noise:
		return np.squeeze(points + noise), np.squeeze(noise)

	return np.squeeze(points + noise)


def jitter_direction_vectors(vectors, sigma=1.5, clip=3, rng1=None, rng2=None, return_noise=False):
	""" Jitter direction """
	assert ((vectors.ndim == 2) or (vectors.ndim == 3))
	assert (vectors.shape[-1] == 3)

	if vectors.ndim == 2:
		vectors = np.expand_dims(vectors, axis=0)

	R_batch = []
	for batch_ind in range(vectors.shape[0]):
		# Create random rotation axes
		if rng1 is not None:
			axes = np.clip(0.5 * rng1.randn(vectors.shape[1], vectors.shape[2]), -1, 1)
		else:
			axes = np.clip(0.5 * np.random.randn(vectors.shape[1], vectors.shape[2]), -1, 1)

		# Get random angles
		if rng2 is not None:
			angles = np.clip(sigma * rng2.randn(vectors.shape[1]), -clip, clip)
		else:
			angles = np.clip(sigma * np.random.randn(vectors.shape[1]), -clip, clip)

		# Get rotation matrices
		R = rotations3D.axisAngleMatrixBatch(angle=angles, rotation_axis=axes)

		vectors[batch_ind] = np.squeeze(np.matmul(R, vectors[batch_ind, ..., np.newaxis]))

		if return_noise:
			R_batch.append(R)

	if return_noise:
		R_batch = np.vstack(R_batch).astype(np.float32)
		R_batch = R_batch.reshape(vectors.shape[0], vectors.shape[1], 3, 3)
		return np.squeeze(vectors), np.squeeze(R_batch)

	return np.squeeze(vectors)
