"""
Use graph cuts for point cloud semantic segmentation and compare against
DGCNN semantic segmentation
"""

import tensorflow as tf
import numpy as np
import importlib
import time
import errno
import os
import sys
import matlab.engine
from scipy import spatial
from progressbar import ProgressBar

# Custom packages
BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))

from PartNet_data import dataset_h5 as partnet_dataset_h5
from utils import evaluation_utils, rotations3D
from evaluation import evaluate_boundary_detection

def get_sem_model(model, model_path, num_point, n_features, n_classes=1, gpu_index=0):
  with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_index)):
      pointclouds_ph = tf.placeholder(tf.float32, shape=(1, num_point, n_features))
      is_training_ph = tf.placeholder(tf.bool, shape=())

      seg_pred = model.get_model(point_cloud=pointclouds_ph, is_training=is_training_ph, normals=None,
								 use_local_frame=False, add_normals=False, bn=True, bn_decay=None,
			  					 align_pointclouds=False, n_classes=n_classes)

      probs = tf.nn.softmax(seg_pred)
      saver = tf.train.Saver()

      # Create a session
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True
      config.allow_soft_placement = True
      config.gpu_options.visible_device_list = str(gpu_index)
      sess = tf.Session(config=config)

      # Restore variables from disk.
      saver.restore(sess, model_path)

      ops = {'pointclouds_ph': pointclouds_ph,
             'is_training_ph': is_training_ph,
             'seg_pred': seg_pred,
             'probs': probs}

  return sess, ops


def sem_inference(sess, ops, pc):
  is_training = False

  # Infer part labels
  feed_dict = {ops["pointclouds_ph"]: pc,
               ops["is_training_ph"]: is_training}

  seg_probs_res, fc_output = sess.run([ops["probs"], ops["seg_pred"]], feed_dict=feed_dict)

  return np.squeeze(seg_probs_res), np.squeeze(fc_output)



if __name__ == '__main__':
	t1 = time.time()

	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
	parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
	parser.add_argument('--category', default='Bag', help='Which single class to test on [default: Bag]')
	parser.add_argument('--semantic_model_path', default='', help='semantic model checkpoint file path')
	parser.add_argument('--boundary_model_path', default='', help='boundary model checkpoint file path')
	parser.add_argument('--dataset', default='', help="Specify a semantic dataset")
	parser.add_argument('--split', default='test', help='Choose which split of the dataset to use [default: test]')
	parser.add_argument('--pairwise_features', default='combine', help='Specify pairwise features for graph cuts [default: combine]')

	ARGS = parser.parse_args()
	print(vars(ARGS))

	# For noise addition
	rng1 = np.random.RandomState(2)
	rng2 = np.random.RandomState(3)
	rng3 = np.random.RandomState(4)

	# Start matlab engine in the background
	future = matlab.engine.start_matlab(async=True)

	# Configuration
	SEMANTIC_MODEL_PATH = ARGS.semantic_model_path
	BOUNDARY_MODEL_PATH = ARGS.boundary_model_path
	GPU_INDEX = ARGS.gpu
	NUM_POINT = ARGS.num_point
	MODEL = importlib.import_module("dgcnn_part_seg")  # import network module
	DATASET_DIR = ARGS.dataset
	CATEGORY = ARGS.category
	SPLIT = ARGS.split

	EVAL_FOLDER = os.path.join("evaluation_metrics", CATEGORY)
	try:
		os.makedirs(EVAL_FOLDER)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

	# Graph cuts settings
	if ARGS.pairwise_features.lower() == 'boundary_confidence':
		PAIRWISE_FEATURES = 'boundary_confidence'
	elif ARGS.pairwise_features.lower() == 'normal_angles':
		PAIRWISE_FEATURES = 'normal_angles'
	elif ARGS.pairwise_features.lower() == 'combine':
		PAIRWISE_FEATURES = 'combine'
	else:
		print("Warning: unknown pairwise features {pt_feat:s}".format(pt_feat=ARGS.pairwise_features))
		print("Boundary confidence will be used instead")
		PAIRWISE_FEATURES = 'boundary_confidence'

	# Import semantic test split
	DATASET = partnet_dataset_h5.Dataset(pc_dataset_dir=DATASET_DIR, category=CATEGORY, split=SPLIT,
										 sampling=False, npoints=-1, rotate_principals_around_normal=True, evaluate=True)
	# Number of segmentation classes
	NUM_SEMANTIC_CLASSES = len(DATASET.segClasses[ARGS.category])

	# Create matlab instance
	eng = future.result()
	# Add current directory to matlab path
	eng.addpath(os.curdir)

	# Gcuts_smoothing
	if PAIRWISE_FEATURES == "combine":
		gcuts_smoothing = [1.0, 1.0]
	else:
		gcuts_smoothing = 1.0


	# Load semantic model
	semantic_sess, semantic_ops = get_sem_model(model=MODEL, model_path=SEMANTIC_MODEL_PATH, num_point=NUM_POINT, n_features=3,
												n_classes=NUM_SEMANTIC_CLASSES, gpu_index=GPU_INDEX)

	# Load model
	boundary_sess, boundary_ops = evaluate_boundary_detection.get_model(model=MODEL, model_path=BOUNDARY_MODEL_PATH,
																		num_point=NUM_POINT, n_features=9, add_normals=True,
																		align_pointclouds=False, rotate_principals_around_normal=True,
																		gpu_index=GPU_INDEX)

	# Init metrics for sem seg model
	# Init mIoU
	shape_iou_tot = 0.0; shape_iou_cnt = 0
	part_intersect = np.zeros((NUM_SEMANTIC_CLASSES), dtype=np.float32)
	part_union = np.zeros((NUM_SEMANTIC_CLASSES), dtype=np.float32)
	# Init chamfer distance
	overall_dcd = 0; exclude_models = 0
	# Init boundary IoU
	boundary_iou = 0.0
	# Init precision, recall metrics
	precision = 0.0; recall = 0.0; f_score = 0.0

	# Init metrics for graph cuts
	# Init mIoU
	shape_iou_tot_graph_cuts = 0.0;	shape_iou_cnt_graph_cuts = 0
	part_intersect_graph_cuts = np.zeros(( NUM_SEMANTIC_CLASSES), dtype=np.float32)
	part_union_graph_cuts = np.zeros((NUM_SEMANTIC_CLASSES), dtype=np.float32)
	# Init chamfer distance
	overall_dcd_graph_cuts = 0.0
	# Init boundary IoU
	boundary_iou_graph_cuts = 0.0
	precision_graph_cuts = 0.0; recall_graph_cuts = 0.0; f_score_graph_cuts = 0.0

	bar = ProgressBar()
	for i in bar(range(len(DATASET))):

		# Get pointclouds from dataset
		points, sem_input_features, sem_pointcloud_features = \
			evaluation_utils.get_pointcloud(dataset=DATASET, idx=i, n_features=3, get_tangent_vectors=False)
		_, boundary_input_features, boundary_pointcloud_features = \
			evaluation_utils.get_pointcloud(dataset=DATASET, idx=i, n_features=9, get_tangent_vectors=True)

		print("Infer part labels and boundaries on point cloud ({current:d}/{total:d}) with {n_points:d} points, from category "
			  "{cat:s} ({dataset:s}) ...".format(current=i + 1, dataset=os.path.basename(DATASET_DIR),
												 n_points=len(points), total=len(DATASET), cat=CATEGORY))

		# Add noise
		points = evaluation_utils.jitter_points(points=points, sigma=0.005, clip=0.01, rng=rng1)
		sem_input_features[:, 0:3] = np.copy(points)
		boundary_input_features[:, 0:3] = np.copy(points)
		sem_pointcloud_features[:, 0:3], R_noise = \
			evaluation_utils.jitter_direction_vectors(sem_pointcloud_features[:, 0:3], sigma=1.5, clip=3, rng1=rng2,
													  rng2=rng3, return_noise=True)
		boundary_pointcloud_features[:, 0:3] = np.copy(sem_pointcloud_features[:, 0:3])
		boundary_input_features[:, 3:6] = np.squeeze(np.matmul(R_noise, boundary_input_features[:, 3:6, np.newaxis]))
		boundary_input_features[:, 6:9] = np.squeeze(np.matmul(R_noise, boundary_input_features[:, 6:9, np.newaxis]))

		# Semantic inference
		semantic_probs, semantic_fc_output = sem_inference(sess=semantic_sess, ops=semantic_ops, pc=np.expand_dims(sem_input_features, 0))
		semantic_segp = np.squeeze(np.argmax(semantic_probs[:, 1:], axis=-1) + 1)


		# Boundary inference
		normals = np.expand_dims(boundary_pointcloud_features[:, 0:3], axis=0)
		boundary_probs, boundary_fc_output = evaluate_boundary_detection.inference(sess=boundary_sess, ops=boundary_ops,
																				   pc=np.expand_dims(boundary_input_features, 0),
																				   normals=normals, rotate_principals_around_normal=True)

		# Get data for graph cuts
		seg_points = points
		seg_semantic = sem_pointcloud_features[:, -1]
		segp_semantic = semantic_segp
		seg_normals = sem_pointcloud_features[:, :3]
		seg_boundary_probs = boundary_probs
		seg_sem_fc_output = semantic_probs[:, 1:]

		# Evaluate semantic segmentation performance wrt. shape and part category mIoU
		shape_iou_tot, shape_iou_cnt, part_intersect, part_union \
			= evaluation_utils.mIoU_calculation(seg=seg_semantic, segp=segp_semantic.astype(np.int32),
												shape_iou_tot=shape_iou_tot,
							   					shape_iou_cnt=shape_iou_cnt, part_intersect=part_intersect,
							   					part_union=part_union, num_semantic_classes=NUM_SEMANTIC_CLASSES)

		# Infer boundaries from semantic segmentation
		segp_semantic[seg_semantic == 0] = 0
		inferred_boundary_seg = evaluation_utils.boundary_annotation(points=seg_points, semantic_seg=seg_semantic)
		inferred_boundary_segp = evaluation_utils.boundary_annotation(points=seg_points, semantic_seg=segp_semantic)

		# Check if segmentations have only 1 part or have disconnected parts
		seg_1_parts = np.unique(seg_semantic)
		seg_1_boundaries = np.unique(inferred_boundary_seg)
		if (len(seg_1_parts) == 1) or (len(seg_1_boundaries) == 1):
			# Ground truth segmentation -> exclude from evaluation
			exclude_models += 1
		else:
			seg_2_parts = np.unique(segp_semantic)
			seg_2_boundaries = np.unique(inferred_boundary_segp)
			if (len(seg_2_parts) == 1) or (len(seg_2_boundaries) == 1):
				# Predicted segmentation -> penalize for chamfer distance
				overall_dcd += evaluation_utils.pc_bounding_box_diagonal(seg_points)
			else:
				# Evaluate inferred boundaries wrt. chamfer distance
				dcd = evaluation_utils.chamfer_distance(segmentation1=np.squeeze(inferred_boundary_seg),
													    segmentation2=np.squeeze(inferred_boundary_segp),
													    points=seg_points)
				overall_dcd += dcd

		# Create kd-tree for surface points
		points_KDTree = spatial.cKDTree(seg_points, copy_data=False, balanced_tree=False, compact_nodes=False)
		# Find maximum sampling distance
		nn_dist, _ = points_KDTree.query(seg_points, k=2)
		nn_dist = nn_dist[:, 1]
		max_dist = np.amax(nn_dist)
		# Boundary distance tolerance threshold
		dist = max_dist

		# Calculate precision
		boundary_points_gt = seg_points[np.squeeze(inferred_boundary_seg==1)]
		if len(boundary_points_gt):
			boundary_points_pred = seg_points[np.squeeze(inferred_boundary_segp==1)]
			shape_precision, TP_precision, FP_precision = evaluation_utils.precision(boundary_points_gt=boundary_points_gt,
																					 boundary_points_pred=boundary_points_pred,
																					 dist=dist)
			precision += shape_precision

			# Calculate recall
			shape_recall, TP_recall, FN_recall = evaluation_utils.recall(boundary_points_gt=boundary_points_gt,
																		 boundary_points_pred=boundary_points_pred,
																		 dist=dist)
			recall += shape_recall

			# Calculate bIoU
			try:
				boundary_iou += ((TP_precision + TP_recall) / float(len(boundary_points_gt) + len(boundary_points_pred)))
			except:
				boundary_iou += 0.0

		# Use graph cuts for point cloud semantic segmentation
		semantic_seg_graph_cuts = np.zeros((seg_points.shape[0]))
		boundary_seg_graph_cuts = np.zeros((seg_points.shape[0]))
		print("Semantic segmentation of point cloud ({current:d}/{total:d}) with {nPoints:d} points, from category "
			  "{cat:s} ({dataset:s}), using graph cuts (pairwise term: {pt_feat:s}) ..."
			  .format(current=i + 1, dataset=os.path.basename(DATASET_DIR), nPoints=len(seg_points),
						  total=len(DATASET), cat=CATEGORY, pt_feat=PAIRWISE_FEATURES))

		# Create matlab.arrays
		semantic_data = seg_points
		if PAIRWISE_FEATURES == 'boundary_confidence':
			semantic_data = np.hstack((semantic_data, seg_boundary_probs[..., np.newaxis]))
			pt_arg = 'max'
		elif PAIRWISE_FEATURES == 'normal_angles':
			semantic_data = np.hstack((semantic_data, seg_normals))
			pt_arg = ''
		elif PAIRWISE_FEATURES == 'combine':
			semantic_data = np.hstack((semantic_data, seg_normals, seg_boundary_probs[..., np.newaxis]))
			pt_arg = 'max'
		else:
			print("ERROR: undefined pairwise feature {pt_feat:s}" .format(pt_feat=PAIRWISE_FEATURES))
			exit(-1)

		point_cloud_data = matlab.double([[p for p in row] for row in semantic_data])
		sem_fc_output = matlab.double([[p for p in fc_out] for fc_out in seg_sem_fc_output])

		# Call point_cloud_seg
		if PAIRWISE_FEATURES == 'combine':
			gcuts_smoothing_data = matlab.double([gcuts_smoothing[0], gcuts_smoothing[1]])
			matlab_out = eng.point_cloud_seg(point_cloud_data, sem_fc_output, gcuts_smoothing_data, PAIRWISE_FEATURES,
											 pt_arg)
		else:
			matlab_out = eng.point_cloud_seg(point_cloud_data, sem_fc_output, gcuts_smoothing, PAIRWISE_FEATURES,
											 pt_arg)
		semantic_seg_graph_cuts = np.squeeze(np.asarray(matlab_out))+1

		# Evaluate graph cuts semantic segmentation performance wrt. shape and part category mIoU
		shape_iou_tot_graph_cuts, shape_iou_cnt_graph_cuts, part_intersect_graph_cuts, part_union_graph_cuts \
			= evaluation_utils.mIoU_calculation(seg=seg_semantic, segp=semantic_seg_graph_cuts.astype(np.int32),
							   shape_iou_tot=shape_iou_tot_graph_cuts, shape_iou_cnt=shape_iou_cnt_graph_cuts,
							   part_intersect=part_intersect_graph_cuts, part_union=part_union_graph_cuts,
							   num_semantic_classes=NUM_SEMANTIC_CLASSES)

		# Infer boundaries from semantic segmentation
		semantic_seg_graph_cuts[seg_semantic == 0] = 0
		inferred_boundary_seg_graph_cuts = evaluation_utils.boundary_annotation(points=seg_points,
																				semantic_seg=semantic_seg_graph_cuts)
		boundary_seg_graph_cuts = np.squeeze(inferred_boundary_seg_graph_cuts)

		penalize = False
		if (len(seg_1_parts) == 1) or (len(seg_1_boundaries) == 1):
			# Ground truth segmentation -> exclude from evaluation
			pass
		else:
			seg_2_parts = np.unique(semantic_seg_graph_cuts)
			seg_2_boundaries = np.unique(inferred_boundary_seg_graph_cuts)
			if (len(seg_2_parts) == 1) or (len(seg_2_boundaries) == 1):
				# Use unary term only
				inferred_boundary_seg_graph_cuts = inferred_boundary_segp
				seg_2_boundaries = np.unique(inferred_boundary_seg_graph_cuts)
				if len(seg_2_boundaries) == 1:
					# Predicted segmentation -> penalize for chamfer distance
					overall_dcd_graph_cuts += evaluation_utils.pc_bounding_box_diagonal(seg_points)
					penalize = True
			if not penalize:
				# Calculate chamfer distance
				dcd = evaluation_utils.chamfer_distance(segmentation1=inferred_boundary_seg,
													    segmentation2=inferred_boundary_seg_graph_cuts,
													    points=seg_points)
				overall_dcd_graph_cuts += dcd

			if len(boundary_points_gt):
				# Calculate precision
				boundary_points_pred_graph_cuts = seg_points[np.squeeze(inferred_boundary_seg_graph_cuts == 1)]
				shape_precision_graph_cuts, TP_precision_graph_cuts, FP_precision_graph_cuts = \
					evaluation_utils.precision(boundary_points_gt=boundary_points_gt,
											   boundary_points_pred=boundary_points_pred_graph_cuts, dist=dist)
				precision_graph_cuts += shape_precision_graph_cuts

				# Calculate recall
				shape_recall_graph_cuts, TP_recall_graph_cuts, FN_recall_graph_cuts = \
					evaluation_utils.recall(boundary_points_gt=boundary_points_gt,
											boundary_points_pred=boundary_points_pred_graph_cuts, dist=dist)
				recall_graph_cuts += shape_recall_graph_cuts

				# Calculate bIoU
				try:
					boundary_iou_graph_cuts += ((TP_precision_graph_cuts + TP_recall_graph_cuts) /
													 float(len(boundary_points_gt) + len(boundary_points_pred_graph_cuts)))
				except:
					boundary_iou_graph_cuts += 0.0

	# Close session
	semantic_sess.close()
	boundary_sess.close()

	# Stop matlab engine
	eng.quit()

	# Log evaluation
	total_seen = len(DATASET)
	buf = "DGCNN\n----------\n"
	# Log precision
	precision /= float(total_seen - exclude_models)
	buf += 'Precision: %f\n' % (precision)
	# Log recall
	recall /= float(total_seen - exclude_models)
	buf += 'Recall: %f\n' % (recall)
	# Log f-score
	try:
		f_score = 2.0 * (precision * recall) / (precision + recall)
	except ZeroDivisionError:
		f_score = 0.0
	buf += 'F-score: %f\n' % (f_score)
	# Log category mIoU
	part_iou = np.nan_to_num(np.divide(part_intersect[1:], part_union[1:]))
	mean_part_iou = np.mean(part_iou)
	buf += 'Category mean IoU: %f\n' % (mean_part_iou)
	# Log shape mIoU
	buf += 'Shape mean IoU: %f\n' % (shape_iou_tot / float(shape_iou_cnt))
	# Log chamfer distance
	try:
		mean_dcd = overall_dcd / float(total_seen - exclude_models)
	except ZeroDivisionError:
		mean_dcd = 1e5
	buf += 'Chamfer distance: %f\n' % (mean_dcd)
	# Log boundary IoU
	boundary_iou /= float(total_seen - exclude_models)
	buf += 'Boundary IoU: %f\n' % (boundary_iou)
	out_list = ['%3.1f' % (item * 100) for item in part_iou.tolist()]
	buf += '%3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f ;%s\n' % (mean_part_iou * 100,
		shape_iou_tot * 100 / float(shape_iou_cnt), mean_dcd * 100, boundary_iou * 100, precision * 100, recall * 100,
		f_score * 100, '[' + ', '.join(out_list) + ']')
	buf += '=SPLIT("%3.1f,%3.1f,%3.1f,%3.1f,%3.1f,%3.1f,%3.1f", ",")\n' % (mean_part_iou * 100,
		shape_iou_tot * 100 / float(shape_iou_cnt), mean_dcd * 100, boundary_iou * 100, precision * 100, recall * 100,
		f_score * 100)

	# Log graph cuts evaluation
	buf += 'Graph cuts\n----------\n'
	buf += PAIRWISE_FEATURES+'\n'
	# Log precision
	precision_graph_cuts /= float(total_seen - exclude_models)
	buf += 'Precision: %f\n' % (precision_graph_cuts)
	# Log recall
	recall_graph_cuts /= float(total_seen - exclude_models)
	buf += 'Recall: %f\n' % (recall_graph_cuts)
	# Log f-score
	try:
		f_score_graph_cuts = 2.0 * (precision_graph_cuts * recall_graph_cuts) / (precision_graph_cuts + recall_graph_cuts)
	except ZeroDivisionError:
		f_score_graph_cuts = 0.0
	buf += 'F-score: %f\n' % (f_score_graph_cuts)
	# Log category mIoU
	part_iou_graph_cuts = np.nan_to_num(np.divide(part_intersect_graph_cuts[1:], part_union_graph_cuts[1:]))
	mean_part_iou_graph_cuts = np.mean(part_iou_graph_cuts)
	buf += 'Category mean IoU: %f\n' % (mean_part_iou_graph_cuts)
	# Log shape mIoU
	buf += 'Shape mean IoU: %f\n' % (shape_iou_tot_graph_cuts / float(shape_iou_cnt_graph_cuts))
	# Log chamfer distance
	try:
		mean_dcd_graph_cuts = overall_dcd_graph_cuts / float(total_seen - exclude_models)
	except ZeroDivisionError:
		mean_dcd_graph_cuts = 1e5
	buf += 'Chamfer distance: %f\n' % (mean_dcd_graph_cuts)
	# Log boundary IoU
	boundary_iou_graph_cuts /= float(total_seen - exclude_models)
	buf += 'Boundary IoU: %f\n' % (boundary_iou_graph_cuts)
	out_list_graph_cuts = ['%3.1f' % (item * 100) for item in part_iou_graph_cuts.tolist()]
	buf += '%3.1f %3.1f;%s\n' % (mean_part_iou_graph_cuts * 100,
								 shape_iou_tot_graph_cuts * 100 / float(shape_iou_cnt_graph_cuts),
								 '[' + ', '.join(out_list_graph_cuts) + ']')
	buf += '%3.1f %3.1f %3.1f %3.1f %3.1f %3.1f %3.1f ;%s\n' % (mean_part_iou_graph_cuts * 100,
		shape_iou_tot_graph_cuts * 100 / float(shape_iou_cnt_graph_cuts), mean_dcd_graph_cuts * 100,
		boundary_iou_graph_cuts * 100, precision_graph_cuts * 100, recall_graph_cuts * 100,
		f_score_graph_cuts * 100, '[' + ', '.join(out_list) + ']')
	buf += '=SPLIT("%3.1f,%3.1f,%3.1f,%3.1f,%3.1f,%3.1f,%3.1f", ",")\n' % (mean_part_iou_graph_cuts * 100,
		shape_iou_tot_graph_cuts * 100 / float(shape_iou_cnt_graph_cuts), mean_dcd_graph_cuts * 100,
		boundary_iou_graph_cuts * 100, precision_graph_cuts * 100, recall_graph_cuts * 100,
		f_score_graph_cuts * 100)

	out_fn = os.path.join(EVAL_FOLDER, "evaluation_metrics_" + PAIRWISE_FEATURES + ".txt")
	with open(out_fn, 'w') as fout:
		fout.write(buf)


	elapsed_time = time.time() - t1
	print("\nFinish evaluation -- time passed {hours:d}:{minutes:d}:{seconds:d}"
		  .format(hours=int((elapsed_time / 60 ** 2) % (60 ** 2)), minutes=int((elapsed_time / 60) % (60)),
				  seconds=int(elapsed_time % 60)))