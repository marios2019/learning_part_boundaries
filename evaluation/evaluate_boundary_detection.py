"""
Evaluate boundary detection w.r.t. precision, recall, f-score, boundary IoU and Chamfer Distance
"""

import tensorflow as tf
import numpy as np
import importlib
import time
import errno
import os
import sys
from scipy import spatial
from progressbar import ProgressBar


# Import custom packages
BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
from ABC_data import dataset_evaluation_h5
from utils import evaluation_utils


def get_model(model, model_path, num_point, n_features, add_normals, align_pointclouds=True,
              rotate_principals_around_normal=True, gpu_index=0):
  batch_size = 1
  with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_index)):
      pointclouds_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, n_features))
      normals_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3), name='normals_ph')
      is_training_ph = tf.placeholder(tf.bool, shape=())

      seg_pred = model.get_model(point_cloud=pointclouds_ph, is_training=is_training_ph, normals=normals_ph,
                                 use_local_frame=True, add_normals=add_normals, bn=True, bn_decay=None,
                                 align_pointclouds=align_pointclouds, n_classes=1)

      seg_pred_ph = None
      if not rotate_principals_around_normal:
        seg_pred_ph = tf.placeholder(tf.float32, shape=(batch_size, num_point, 1), name='seg_pred_ph')
        # Concatenate features from 2 branches of the network
        concat_seg_pred = tf.concat([seg_pred, seg_pred_ph], axis=-1)
        # Max-pool
        max_seg_pred = tf.reduce_max(concat_seg_pred, axis=-1)
        probs = tf.nn.sigmoid(max_seg_pred)
      else:
        probs = tf.nn.sigmoid(seg_pred)
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
             'normals_ph': normals_ph,
             'seg_pred': seg_pred,
             'seg_pred_ph': seg_pred_ph,
             'probs': probs}

  return sess, ops


def inference(sess, ops, pc, normals, rotate_principals_around_normal=False):
  is_training = False

  # Infer part labels
  feed_dict = {ops["pointclouds_ph"]: pc,
               ops['normals_ph']: normals,
               ops["is_training_ph"]: is_training}

  if not rotate_principals_around_normal:
    # Use 2-RoSy field
    seg_pred_1 = sess.run(ops["seg_pred"], feed_dict=feed_dict)
    seg_pred_1 = seg_pred_1
    # Invert principal directions
    pc[..., -6:] = -pc[..., -6:]
    feed_dict[ops["pointclouds_ph"]] = pc
    feed_dict[ops["seg_pred_ph"]] = seg_pred_1
    seg_probs_res, seg_pred_2 = sess.run([ops["probs"], ops["seg_pred"]], feed_dict=feed_dict)
    fc_output = np.concatenate((seg_pred_1, seg_pred_2), axis=-1)
    fc_output = np.amax(fc_output, axis=-1)
  else:
    seg_probs_res, fc_output = sess.run([ops["probs"], ops["seg_pred"]], feed_dict=feed_dict)

  return np.squeeze(seg_probs_res), np.squeeze(fc_output)


if __name__ == '__main__':
  t1 = time.time()

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
  parser.add_argument('--model_path', default='', help='model checkpoint file path')
  parser.add_argument('--dataset', default='', help="Specify dataset [default: '']")
  parser.add_argument('--split', default='test', help='Choose which split of the dataset to use [default: test]')
  parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
  parser.add_argument('--conf_threshold', type=float, default=0.63, help="Confidence threshold [default: 0.63]")
  parser.add_argument('--radius', type=float, default=1, help='Ball radius for adaptive metrics [default: 1]')

  ARGS = parser.parse_args()
  print(vars(ARGS))

  # Configuration
  MODEL_PATH = ARGS.model_path
  GPU_INDEX = ARGS.gpu
  MODEL = importlib.import_module("dgcnn_part_seg")  # import network module
  DATASET_NAME = os.path.basename(ARGS.dataset.lower())
  NUM_POINT = ARGS.num_point
  CONF_THRESHOLD = ARGS.conf_threshold
  RADIUS = ARGS.radius
  DATASET_DIR = ARGS.dataset
  SPLIT = ARGS.split
  ROTATE_PRINCIPALS_AROUND_NORMAL = True
  ADD_NORMALS=True
  ALIGN_POINTCLOUDS = True
  N_FEATURES = 9

  EVAL_FOLDER = os.path.join("evaluation_metrics", "radius_"+str(RADIUS))
  try:
    os.makedirs(EVAL_FOLDER)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise

  # Load model
  sess, ops = get_model(model=MODEL, model_path=MODEL_PATH, num_point=NUM_POINT, n_features=N_FEATURES, add_normals=ADD_NORMALS,
                        align_pointclouds=True, rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL, gpu_index=GPU_INDEX)

  # Import evaluation dataset
  DATASET = dataset_evaluation_h5.Dataset(pc_dataset_dir=DATASET_DIR, split=SPLIT, sampling=False, npoints=NUM_POINT,
                                          rotate_principals_around_normal= ROTATE_PRINCIPALS_AROUND_NORMAL)


  # Initialize evaluation metrics variables
  precision = 0.0; recall = 0.0; f_score = 0.0; overall_dcd = 0; exclude_models = 0; boundary_iou = 0.0

  # For noise addition
  rng1 = np.random.RandomState(2)
  rng2 = np.random.RandomState(3)
  rng3 = np.random.RandomState(4)

  bar = ProgressBar()
  for i in bar(range(len(DATASET))):

    # Get pointclouds from dataset
    points, input_features, pointcloud_features, boundary_points, boundary_normals = \
          evaluation_utils.get_pointcloud_evaluation(dataset=DATASET, idx=i, n_features=N_FEATURES)
    print("Infer boundaries on point cloud ({current:d}/{total:d}) with {n_points:d} points, from {dataset:s}, "
          " using model {model:s}...".format(current=i + 1, dataset=os.path.basename(DATASET_NAME),n_points=len(points),
                                             total=len(DATASET), model=MODEL_PATH))

    # Add noise
    # Jitter points
    points = evaluation_utils.jitter_points(points=points, sigma=0.005, clip=0.01, rng=rng1)
    input_features[:, 0:3] = np.copy(points)
    # Jitter normals
    pointcloud_features[:, 0:3], R = evaluation_utils.jitter_direction_vectors(pointcloud_features[:, 0:3],
                                                                              sigma=1.5, clip=3, rng1=rng2,
                                                                              rng2=rng3, return_noise=True)
    # Jitter tangent vectors
    input_features[:, 3:6] = np.squeeze(np.matmul(R, input_features[:, 3:6, np.newaxis]))
    input_features[:, 6:9] = np.squeeze(np.matmul(R, input_features[:, 6:9, np.newaxis]))

    # Boundary inference
    normals = np.expand_dims(pointcloud_features[:, 0:3], axis=0)
    probs, fc_output = inference(sess=sess, ops=ops, pc=np.expand_dims(input_features, 0), normals=normals,
                                 rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL)

    # Threshold boundary confidence
    segp = np.zeros((NUM_POINT,1), dtype=np.float32)
    segp[probs > CONF_THRESHOLD] = 1

    # Evaluate boundary detection performance
    # Create kd-tree for surface points
    points_KDTree = spatial.cKDTree(points, copy_data=False, balanced_tree=False, compact_nodes=False)
    # Find maximum sampling distance
    nn_dist, _ = points_KDTree.query(points, k=2)
    nn_dist = nn_dist[:, 1]
    max_dist = np.amax(nn_dist)
    # Boundary tolerance threshold
    dist = RADIUS * max_dist

    # Calculate precision
    boundary_points_pred = points[np.squeeze(segp==1)]
    shape_precision, TP_precision, FP_precision = evaluation_utils.precision(boundary_points_gt=boundary_points,
                                                                             boundary_points_pred=boundary_points_pred,
                                                                             dist=dist)
    precision += shape_precision

    # Calculate recall
    shape_recall, TP_recall, FP_recall = evaluation_utils.recall(boundary_points_gt=boundary_points,
                                                                 boundary_points_pred=boundary_points_pred,
                                                                 dist=dist)
    recall += shape_recall

    # Calculate bIoU
    try:
      boundary_iou += ((TP_precision + TP_recall) / float(len(boundary_points) + len(boundary_points_pred)))
    except:
      boundary_iou += 0.0


    if len(boundary_points) == 0:
        # Ground truth segmentation -> exclude from evaluation
        exclude_models += 1
    else:
       seg_2_boundaries = np.unique(segp)
       if len(seg_2_boundaries) == 1:
         # Predicted segmentation -> penalize for chamfer distance
         overall_dcd += evaluation_utils.pc_bounding_box_diagonal(points)
       else:
         # Evaluate inferred boundaries wrt. chamfer distance
         dcd = evaluation_utils.chamfer_distance(segmentation1_points=boundary_points,
                                                 segmentation2_points=points[np.squeeze(segp==1)],
                                                 points=points)
         overall_dcd += dcd

  # Log network evaluatioo
  total_seen = len(DATASET)
  # Log precision
  precision /= float(total_seen - exclude_models)
  buf = 'Precision: %f\n' % (precision)
  # Log recall
  recall /= float(total_seen)
  buf += 'Recall: %f\n' % (recall - exclude_models)
  # Log f-score
  try:
    f_score = 2.0 * (precision * recall) / (precision + recall)
  except ZeroDivisionError:
    f_score = 0.0
  buf += 'F-score: %f\n' % (f_score)
  # Log chamfer distance
  try:
    mean_dcd = overall_dcd / float(total_seen - exclude_models)
  except ZeroDivisionError:
    mean_dcd = 1e5
  buf += 'Chamfer distance: %f\n' % (mean_dcd)
  boundary_iou /= float(total_seen - exclude_models)
  buf += 'Boundary IoU: %f\n' % (boundary_iou)
  buf += '%f,%f,%f,%f,%f\n' % (precision, recall, f_score, mean_dcd, boundary_iou)
  buf += '=SPLIT("%3.1f,%3.1f,%3.1f,%3.1f,%3.1f", ",")\n' % (precision*100, recall*100, f_score*100, mean_dcd*100, boundary_iou*100)
  print(buf)
  with open(os.path.join(EVAL_FOLDER, "evaluation_metrics.txt"), 'w') as fout:
      fout.write(buf)

  elapsed_time = time.time() - t1
  print("\nFinish evaluation -- time passed {hours:d}:{minutes:d}:{seconds:d}"
        .format(hours=int((elapsed_time / 60 ** 2) % (60 ** 2)), minutes=int((elapsed_time / 60) % (60)),
                seconds=int(elapsed_time % 60)))
