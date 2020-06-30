import argparse
import math
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import pickle
import os
import errno
import sys
from scipy import spatial

current_milli_time = lambda: time.time() * 1000.0

# Import custom packages
BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))
from utils import evaluation_utils
from PartNet_data import dataset_h5 as partnet_dataset_h5
from ABC_data import dataset_h5 as abc_dataset_h5
from ABC_data import dataset_evaluation_h5 as abc_dataset_evaluation_h5

TOWER_NAME = 'tower'

# DEFAULT SETTINGS
parser = argparse.ArgumentParser()
parser.add_argument('--num_gpu', type=int, default=1, help='The number of GPUs to use [default: 1]')
parser.add_argument('--dataset', default='', help="Desired dataset [default: '']")
parser.add_argument('--output_dir', type=str, default='experiment', help='Directory that stores all training logs and trained models')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--nepoch', type=int, default=100, help='Epoch to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--drop_prob', type=float, default=0.4, help='Dropout [default: 0.4]')
parser.add_argument('--decay_step', type=int, default=10, help='Decay step [default: 10]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate [default: 0.5]')
parser.add_argument('--batch_norm', action='store_true', help="Use of batch normalization [default: False]")
parser.add_argument('--align_pointclouds', action='store_true', help='Global alignment of point clouds')
parser.add_argument('--category', default='', help="Dataset category [default: '']")
parser.add_argument('--conf_threshold', type=float, default=0.5, help="Confidence threshold [default: 0.5]")
parser.add_argument('--abc_dataset', action='store_true', help='Use ABC dataset for training instead of PartNet')
FLAGS = parser.parse_args()

# Configuration
NUM_GPU = FLAGS.num_gpu
OPTIMIZER = FLAGS.optimizer
BATCH_NORM = FLAGS.batch_norm
ALIGN_POINTCLOUDS = FLAGS.align_pointclouds
ROTATE_PRINCIPALS_AROUND_NORMAL = True
CATEGORY = FLAGS.category
DATASET_DIR = FLAGS.dataset
ABC_DATASET = FLAGS.abc_dataset

# Configure input features
n_features = 9 # use tangent vectors to construct local frame
ADD_NORMALS = True

# Hyperparameters
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.nepoch
BASE_LEARNING_RATE = FLAGS.learning_rate
LEARNING_RATE_CLIP = 1e-5
MOMENTUM = FLAGS.momentum
DROP_PROB = FLAGS.drop_prob
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = FLAGS.decay_rate
BN_DECAY_DECAY_STEP = float(FLAGS.decay_step)
BN_DECAY_CLIP = 0.99
CONF_THRESHOLD = FLAGS.conf_threshold

# Check NUM_GPU
assert(not BATCH_SIZE % NUM_GPU)

# Import network module
MODEL = importlib.import_module("dgcnn_part_seg")  # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', "dgcnn_part_seg.py")

# Create folders for trained models, logs and summaries
def create_dir(DIR):
  try:
    os.makedirs(DIR)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise
OUTPUT_DIR = FLAGS.output_dir
create_dir(DIR=OUTPUT_DIR)
MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'trained_models')
create_dir(DIR=MODEL_STORAGE_PATH)
TRAINED_MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'trained_model')
create_dir(DIR=TRAINED_MODEL_STORAGE_PATH)
LOG_STORAGE_PATH = os.path.join(OUTPUT_DIR, 'logs')
create_dir(DIR=LOG_STORAGE_PATH)
SUMMARIES_FOLDER =  os.path.join(OUTPUT_DIR, 'summaries')
create_dir(DIR=SUMMARIES_FOLDER)
BACKUP_MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, "backup_model")
create_dir(DIR=BACKUP_MODEL_STORAGE_PATH)
ACC_BEST_MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, "acc_best_model")
create_dir(DIR=ACC_BEST_MODEL_STORAGE_PATH)
DCD_BEST_MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, "dcd_best_model")
create_dir(DIR=DCD_BEST_MODEL_STORAGE_PATH)
LOSS_BEST_MODEL_STORAGE_PATH = os.path.join(OUTPUT_DIR, "loss_best_model")
create_dir(DIR=LOSS_BEST_MODEL_STORAGE_PATH)
os.system('cp %s %s' % (MODEL_FILE, LOG_STORAGE_PATH))  # bkp of model def
os.system('cp train_multi_gpu.py %s' % (LOG_STORAGE_PATH))  # bkp of train procedure
if not os.path.isfile(os.path.join(LOG_STORAGE_PATH, 'log_train.txt')):
  flog = open(os.path.join(LOG_STORAGE_PATH, 'log_train.txt'), 'w')
  flog.write("Begin new training\n")
  flog.write(str(FLAGS) + '\n')
  flog.flush()
else:
  flog = open(os.path.join(LOG_STORAGE_PATH, 'log_train.txt'), 'a')
  flog.write("Continue training\n")
  flog.write(str(FLAGS) + '\n')
  flog.flush()

# epoch log
EPOCH_DICT = {"cur_epoch": 0, "cur_train_batch": 0, "train_idxs": [], "train_flag": True, "best_model_acc": 0.0,
              "best_model_dcd": sys.float_info.max, "best_model_loss": sys.float_info.max}
EPOCH_PKL = "epoch.pkl"

# Load data
# Dataset training and validation split
if ABC_DATASET:
  TRAIN_DATASET = abc_dataset_h5.Dataset(pc_dataset_dir=DATASET_DIR, split='train', sampling=True,
                                         npoints=NUM_POINT, rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL,
                                         return_weights=True)
  VAL_DATASET = abc_dataset_evaluation_h5.Dataset(pc_dataset_dir=os.path.join(os.path.dirname(DATASET_DIR), "boundary_seg_poisson_10K_evaluation_h5"),
                                                  split='val', sampling=True, npoints=NUM_POINT,
                                                  rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL)
else:
  TRAIN_DATASET = partnet_dataset_h5.Dataset(pc_dataset_dir=DATASET_DIR, split='train', sampling=True,
                                             npoints=NUM_POINT, rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL,
                                             category=CATEGORY, return_weights=True)
  VAL_DATASET = partnet_dataset_h5.Dataset(pc_dataset_dir=DATASET_DIR, split='val', sampling=True,
                                           npoints=NUM_POINT, rotate_principals_around_normal=ROTATE_PRINCIPALS_AROUND_NORMAL,
                                           category=CATEGORY, return_weights=True)


def printout(flog, data):
  print(data)
  flog.write(data + '\n')
  flog.flush()


def save_epoch_dict():
  pkl_fn = os.path.join(LOG_STORAGE_PATH, EPOCH_PKL)
  with open(pkl_fn+'.tmp', "wb") as f:
    pickle.dump(EPOCH_DICT, f)
  os.rename(pkl_fn+'.tmp', pkl_fn)


# Reduce learning rate if for a period of time accuracy does not increase
learning_rate_np = FLAGS.learning_rate
best_metric = 0.0; wait = 0
monitor_op = lambda a, b, op, min_delta: op(a, b + min_delta)
def reduce_lr_on_plateau(cur_metric, factor=0.5, patience=10, min_lr=1e-5, min_delta=1e-4, mode='max'):
  global learning_rate_np, best_metric, wait

  # Check if current accuracy > best accuracy
  if mode.lower() == 'max':
    op = np.greater
    # Initialize best metric
    if EPOCH_DICT["cur_epoch"] == 0:
      best_metric = 0.0
  # Check if current loss  < best loss
  elif mode.lower() == 'min':
    op = np.less
    min_delta = -min_delta
    # Initialize best metric
    if EPOCH_DICT["cur_epoch"] == 0:
      best_metric = np.Inf
  else:
    print("ERROR: unknown mode '%s' for monitoring specified metric" % (mode))
    exit(-1)

  if monitor_op(cur_metric, best_metric, op, min_delta):
    best_metric = cur_metric
    wait = 0

    return 0

  # Wait for #patience epochs while metric is on a plateau and then update learning rate
  wait += 1
  if wait >= patience:
    learning_rate_np = np.maximum(learning_rate_np * factor, min_lr)
    wait = 0

    return 1


def get_bn_decay(batch):
  num_batches = (len(TRAIN_DATASET) + BATCH_SIZE - 1) / BATCH_SIZE

  global_step = batch; decay_step = BN_DECAY_DECAY_STEP * num_batches

  bn_momentum = tf.train.exponential_decay(
    BN_INIT_DECAY,
    global_step,
    decay_step,
    BN_DECAY_DECAY_RATE,
    staircase=True)

  bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

  return bn_decay


def average_gradients(tower_grads):
  """Calculate average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been 
     averaged across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      if g is None:
        continue
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)

  return average_grads


def cond_export_model(curr_metric, best_metric, metrics, cond, directory, epoch, saver, sess):
  """ Store model if condition is satisfied """
  assert(cond.lower() == "max" or cond.lower() == "min")

  status = best_metric < curr_metric if cond.lower() == "max" else best_metric > curr_metric
  if status:
    saver.save(sess, os.path.join(directory, "model.ckpt"))
    best_metric = curr_metric
    # Log evaluation metrics
    with open(os.path.join(directory, "evaluation_metrics.txt"), 'w') as f:
      f.write('---- EPOCH %03d EVALUATION ----\n' % (epoch))
      f.write('eval accuracy: %f\n' % (metrics[0]))
      f.write('eval dcd: %f\n' % (metrics[1]))
      f.write('eval loss: %f\n' % (metrics[2]))

  return best_metric


def train():
  global EPOCH_DICT, learning_rate_np

  # Save a snapshot of the model each x epochs
  save_x_epochs = 10

  with tf.Graph().as_default(), tf.device('/cpu:0'):

    # Note the global_step=batch parameter to minimize.
    # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
    batch = tf.Variable(0, trainable=False)

    # Dynamic lr scheduling
    learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate_ph")

    # Batch normalization
    if BATCH_NORM:
      bn_decay = get_bn_decay(batch)
      bn_decay_op = tf.summary.scalar('bn_decay', bn_decay)
    else:
      bn_decay = None
      bn_decay_op = None

    lr_op = tf.summary.scalar('learning_rate', learning_rate)
    batch_op = tf.summary.scalar('batch_number', batch)

    # Get training operator
    print("--- Get training operator")
    if OPTIMIZER == 'momentum':
      trainer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
    elif OPTIMIZER == 'adam':
      trainer = tf.train.AdamOptimizer(learning_rate)
    else:
      print("Error: unknown optimizer {opt:s}".format(opt=OPTIMIZER))
      exit(-1)

    # store tensors for different gpus
    tower_grads = []; pointclouds_phs = []; input_label_phs = []; weights_phs = []; normals_phs = []; seg_phs = []
    seg_pred_phs = []; seg_pred_list = []; conf_list = []; is_training_phs = []

    with tf.variable_scope(tf.get_variable_scope()):
      for i in xrange(NUM_GPU):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
            # Create placeholders
            # for points
            pointclouds_phs.append(tf.placeholder(tf.float32, shape=(BATCH_SIZE / NUM_GPU, NUM_POINT, n_features),
                                                  name='pointclouds_ph'))
            # for part labels
            seg_phs.append(tf.placeholder(tf.float32, shape=(BATCH_SIZE / NUM_GPU, NUM_POINT, 1), name='seg_ph'))
            is_training_phs.append(tf.placeholder(tf.bool, shape=(), name='is_training_ph'))
            # Use to construct local frame for each central point
            normals_phs.append(tf.placeholder(tf.float32, shape=(BATCH_SIZE / NUM_GPU, NUM_POINT, 3), name='normals_ph'))
            # for cross-entropy weights
            weights_phs.append(tf.placeholder(tf.float32, shape=(BATCH_SIZE / NUM_GPU, 1), name='weights_ph'))

            # Get model and loss
            print("--- Get model and loss")
            seg_pred = MODEL.get_model(point_cloud=pointclouds_phs[-1], is_training=is_training_phs[-1],
                                       normals=normals_phs[-1], add_normals=ADD_NORMALS, bn=BATCH_NORM, bn_decay=bn_decay,
                                       align_pointclouds=ALIGN_POINTCLOUDS, drop_prob=DROP_PROB, n_classes=1)
            seg_pred_list.append(seg_pred)

            if not ROTATE_PRINCIPALS_AROUND_NORMAL: # Use principal vector instead of random tangent vectors
              seg_pred_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE / NUM_GPU, NUM_POINT, 1), name='seg_pred_ph')
              # Concatenate features from 2 branches of the network
              concat_seg_pred = tf.concat([seg_pred, seg_pred_ph], axis=-1)
              # Max-pool
              max_seg_pred = tf.reduce_max(concat_seg_pred, axis=-1, keep_dims=True)
              # Get boundary confidence
              conf = tf.nn.sigmoid(max_seg_pred)
              conf_list.append(conf)
              loss, per_instance_seg_loss = MODEL.get_loss(seg_pred=max_seg_pred, seg=seg_phs[-1], weights=weights_phs[-1])
              seg_pred_phs.append(seg_pred_ph)
            else:
              # Get boundary confidence
              conf = tf.nn.sigmoid(seg_pred)
              conf_list.append(conf)
              loss, per_instance_seg_loss = MODEL.get_loss(seg_pred=seg_pred, seg=seg_phs[-1], weights=weights_phs[-1])

            # Get predicted boundaries based on network's boundary confidence
            comparison = tf.greater_equal(conf, tf.constant(CONF_THRESHOLD))
            per_instance_seg_pred_res = tf.where(comparison, tf.ones(conf.shape, dtype=tf.float32), tf.zeros(conf.shape, dtype=tf.float32))

            # Create placeholders and summaries for metrics
            total_training_loss_ph = tf.placeholder(tf.float32, shape=(), name='total_training_loss_ph')
            total_testing_loss_ph = tf.placeholder(tf.float32, shape=(), name='total_testing_loss_ph')

            seg_training_acc_ph = tf.placeholder(tf.float32, shape=(), name='seg_training_acc_ph')
            seg_testing_acc_ph = tf.placeholder(tf.float32, shape=(), name='seg_testing_acc_ph')

            total_testing_dcd_ph = tf.placeholder(tf.float32, shape=(), name='total_testing_dcd_ph')

            total_train_loss_sum_op = tf.summary.scalar('total_training_loss', total_training_loss_ph)
            total_test_loss_sum_op = tf.summary.scalar('total_testing_loss', total_testing_loss_ph)

            seg_train_acc_sum_op = tf.summary.scalar('seg_training_acc', seg_training_acc_ph)
            seg_test_acc_sum_op = tf.summary.scalar('seg_testing_acc', seg_testing_acc_ph)

            total_test_dcd_sum_op = tf.summary.scalar('total_testing_dcd', total_testing_dcd_ph)

            # Variables sharing
            tf.get_variable_scope().reuse_variables()

            # Compute gradients
            grads = trainer.compute_gradients(loss)
            tower_grads.append(grads)

    # Calculate average gradient for each shared variable
    grads = average_gradients(tower_grads)

    # Backpropagation
    train_op = trainer.apply_gradients(grads, global_step=batch)

    # Save variables
    saver = tf.train.Saver(tf.global_variables(), sharded=True, max_to_keep=20)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    if FLAGS.num_gpu == 1:
      config.gpu_options.visible_device_list = '0'
    sess = tf.Session(config=config)

    if os.path.isfile(os.path.join(TRAINED_MODEL_STORAGE_PATH, "checkpoint")) \
            and os.path.isfile(os.path.join(LOG_STORAGE_PATH, EPOCH_PKL)):
      # Restore variables from disk
      saver.restore(sess, os.path.join(TRAINED_MODEL_STORAGE_PATH, "model.ckpt"))
      with open(os.path.join(LOG_STORAGE_PATH, EPOCH_PKL), "rb") as f:
        EPOCH_DICT = pickle.load(f)
      # Restore learning rate
      if "cur_learning_rate" in EPOCH_DICT.keys():
        learning_rate_np = EPOCH_DICT["cur_learning_rate"]
      # Move model to backup folder
      os.system("cp -rv {src:s}/* {dst:s}".format(src=TRAINED_MODEL_STORAGE_PATH, dst=BACKUP_MODEL_STORAGE_PATH))
    else:
      # Init variables
      init = tf.global_variables_initializer()
      sess.run(init)

    # Epoch to start
    start_epoch = EPOCH_DICT["cur_epoch"]

    # Operations used for training and testing
    ops = {"tower_grads": tower_grads,
           "pointclouds_phs": pointclouds_phs,
           "input_label_phs": input_label_phs,
           "normals_phs": normals_phs,
           "weights_phs": weights_phs,
           "seg_phs": seg_phs,
           "is_training_phs": is_training_phs,
           "lr_op": lr_op,
           "batch_op": batch_op,
           "bn_decay_op": bn_decay_op,
           "seg_pred_list": seg_pred_list,
           "seg_pred_phs": seg_pred_phs,
           "conf_list": conf_list,
           "loss": loss,
           "per_instance_seg_loss": per_instance_seg_loss,
           "per_instance_seg_pred_res": per_instance_seg_pred_res,
           "total_training_loss_ph": total_training_loss_ph,
           "total_testing_loss_ph": total_testing_loss_ph,
           "seg_training_acc_ph": seg_training_acc_ph,
           "seg_testing_acc_ph": seg_testing_acc_ph,
           "total_testing_dcd_ph": total_testing_dcd_ph,
           "total_train_loss_sum_op": total_train_loss_sum_op,
           "total_test_loss_sum_op": total_test_loss_sum_op,
           "seg_train_acc_sum_op": seg_train_acc_sum_op,
           "seg_test_acc_sum_op": seg_test_acc_sum_op,
           "total_test_dcd_sum_op": total_test_dcd_sum_op,
           "train_op": train_op,
           'learning_rate_ph': learning_rate}

    train_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(SUMMARIES_FOLDER + '/test')

    fcmd = open(os.path.join(LOG_STORAGE_PATH, 'cmd.txt'), 'w')
    fcmd.write(str(FLAGS))
    fcmd.close()

    # Time train
    train_start_time = time.time()
    for epoch in range(start_epoch, MAX_EPOCH):
      # Time epoch
      epoch_start_time = time.time()

      if EPOCH_DICT["train_flag"]:
        # Train for one epoch
        start_one_epoch_train_time = current_milli_time()
        printout(flog, '\n>>> Training for the epoch %d/%d ...' % (epoch, MAX_EPOCH))
        train_one_epoch(sess=sess, epoch=epoch, ops=ops, writer=train_writer, saver=saver)
        end_one_epoch_train_time = current_milli_time()
        buf = "Epoch {epoch:3d} train time: {train_time:.4f}".format(epoch=epoch,
                                                                     train_time=(end_one_epoch_train_time
                                                                                - start_one_epoch_train_time) / 1000.0)
        printout(flog, buf)

      # Validate training
      printout(flog, '\n<<< Testing on the validation split ...')
      # Return
      #   metrics[0]=curr_test_acc,
      #   metrics[1]=curr_test_dcd,
      #   metrics[2]=curr_test_loss,
      metrics = eval_one_epoch(sess=sess, epoch=epoch, ops=ops, writer=test_writer)

      # Store current learning rate
      EPOCH_DICT["cur_learning_rate"] = learning_rate_np

      # Allow training
      EPOCH_DICT["train_flag"] = True

      # Save variables to disk
      saver.save(sess, os.path.join(TRAINED_MODEL_STORAGE_PATH, "model.ckpt"))

      # Save best model wrt test accuracy
      EPOCH_DICT["best_model_acc"] = cond_export_model(curr_metric=metrics[0], best_metric=EPOCH_DICT["best_model_acc"],
                                                       metrics=metrics, cond="max", directory=ACC_BEST_MODEL_STORAGE_PATH,
                                                       epoch=epoch, saver=saver, sess=sess)
      # Save best model wrt test dcd
      EPOCH_DICT["best_model_dcd"] = cond_export_model(curr_metric=metrics[1], best_metric=EPOCH_DICT["best_model_dcd"],
                                                       metrics=metrics, cond="min", directory=DCD_BEST_MODEL_STORAGE_PATH,
                                                       epoch=epoch, saver=saver, sess=sess)
      # Save best model wrt test loss
      EPOCH_DICT["best_model_loss"] = cond_export_model(curr_metric=metrics[2], best_metric=EPOCH_DICT["best_model_loss"],
                                                        metrics=metrics, cond="min", directory=LOSS_BEST_MODEL_STORAGE_PATH,
                                                        epoch=epoch, saver=saver, sess=sess)

      # Log current epoch
      EPOCH_DICT["cur_epoch"] += 1
      save_epoch_dict()

      # Save snapshots for every x epochs
      if (EPOCH_DICT["cur_epoch"] + 1) % save_x_epochs == 0:
        cp_filename = saver.save(sess, os.path.join(MODEL_STORAGE_PATH, 'epoch_' + str(epoch) + '.ckpt'))
        printout(flog, 'Successfully store the checkpoint model into ' + cp_filename)

      # Elapsed one epoch time
      epoch_elapsed_time = time.time() - epoch_start_time
      buf = "Epoch time passed {hours:d}:{minutes:d}:{seconds:d}".format(
        hours=int((epoch_elapsed_time / 60 ** 2) % (60 ** 2)),
        minutes=int((epoch_elapsed_time / 60) % (60)),
        seconds=int(epoch_elapsed_time % 60))
      printout(flog, buf)

    # Elapsed training time
    train_elapsed_time = time.time() - train_start_time
    buf = "Train time passed {hours:d}:{minutes:d}:{seconds:d}".format(
      hours=int((train_elapsed_time / 60 ** 2) % (60 ** 2)),
      minutes=int((train_elapsed_time / 60) % (60)),
      seconds=int(train_elapsed_time % 60))
    printout(flog, buf)

    flog.close()
##### End training


def get_batch(dataset, idxs, start_idx, end_idx):
  bsize = end_idx - start_idx
  batch_data = np.zeros((bsize, NUM_POINT, n_features), dtype=np.float32)
  batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
  batch_normals = np.zeros((bsize, NUM_POINT, 3), dtype=np.float32)
  batch_weights = np.zeros((bsize, 1), dtype=np.float32)
  for i in range(bsize):
    # Get data
    ps, normals, tangent_vectors, seg, weights = dataset[idxs[i + start_idx]]

    # Assign data
    batch_data[i, :, 0:3] = ps
    batch_data[i, :, 3:9] = tangent_vectors
    batch_normals[i, ...] = normals
    batch_label[i, :] = seg
    batch_weights[i] = np.amax(weights)

  return batch_data, np.expand_dims(batch_label, axis=-1), batch_normals, batch_weights


def get_batch_evaluation(dataset, idxs, start_idx, end_idx):
  bsize = end_idx - start_idx
  batch_data = np.zeros((bsize, NUM_POINT, n_features), dtype=np.float32)
  batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
  batch_normals = np.zeros((bsize, NUM_POINT, 3), dtype=np.float32)
  batch_weights = np.zeros((bsize, 1), dtype=np.float32)
  batch_boundary_points = np.zeros((bsize, NUM_POINT, 3), dtype=np.float32)

  for i in range(bsize):
    # Get data
    ps, normals, tangent_vectors, seg = dataset[idxs[i + start_idx]]
    ind = np.where(seg == 0)[0]
    non_boundary_points = ps[ind]
    non_boundary_normals = normals[ind]
    ind = np.where(seg == 1)[0]
    boundary_points = ps[ind]

    # Assign data
    batch_data[i, :, 0:3] = non_boundary_points
    batch_data[i, :, 3:9] = tangent_vectors
    batch_boundary_points[i] = boundary_points
    batch_normals[i, ...] = non_boundary_normals

    # Create kd-tree for non boundary points
    points_KDTree = spatial.cKDTree(non_boundary_points, copy_data=False, balanced_tree=False, compact_nodes=False)
    # Find maximum sampling distance
    nn_dist, _ = points_KDTree.query(non_boundary_points, k=2)
    nn_dist = nn_dist[:, 1]
    avg_dist = np.mean(nn_dist) # Distance for annotating ground truth boundaries
    # Get ground truth boundaries
    seg = np.zeros((non_boundary_points.shape[0],), dtype=np.float32)
    nn_ind = points_KDTree.query_ball_point(boundary_points, avg_dist)
    for indices in nn_ind:
      if len(indices):
        seg[indices] = 1
    batch_label[i, :] = seg
    batch_weights[i] = np.amax(abc_dataset_evaluation_h5.point_weights(labels=seg))

  return batch_data, np.expand_dims(batch_label, axis=-1), batch_normals, batch_weights, batch_boundary_points


def train_one_epoch(sess, epoch, ops, writer, saver):
  global EPOCH_DICT

  # Train batch to start
  start_train_batch = EPOCH_DICT["cur_train_batch"]

  is_training = True

  num_batches = (len(TRAIN_DATASET) + BATCH_SIZE - 1) / BATCH_SIZE
  if start_train_batch == 0:
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    # Shuffle train samples
    np.random.shuffle(train_idxs)
    EPOCH_DICT["train_idxs"] = train_idxs
  else:
    # Restore train_idxs from previous training
    train_idxs = EPOCH_DICT["train_idxs"]

  # Init variables for accuracy and loss
  total_loss = 0.0; total_seg_acc = 0.0

  # Init data
  batch_data = np.zeros((BATCH_SIZE, NUM_POINT, n_features))
  batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 1)).astype(np.int32)
  batch_normals = np.zeros((BATCH_SIZE, NUM_POINT, 3)).astype(np.float32)
  batch_weights = np.zeros((BATCH_SIZE, 1)).astype(np.float32)

  # Iterate through all batches
  for batch_idx in range(start_train_batch, num_batches):

    start_idx = batch_idx * BATCH_SIZE
    end_idx = min(len(TRAIN_DATASET), (batch_idx + 1) * BATCH_SIZE)
    cur_batch_data, cur_batch_label, cur_batch_normals, cur_batch_weights = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
    cur_batch_size = end_idx - start_idx
    if cur_batch_size == BATCH_SIZE:
      batch_data = cur_batch_data
      batch_label = cur_batch_label
      batch_normals = cur_batch_normals
      batch_weights = cur_batch_weights
    else:  # Last batch might be smaller than the BATCH_SIZE
      batch_data[0:cur_batch_size] = cur_batch_data
      batch_label[0:cur_batch_size] = cur_batch_label
      batch_normals[0:cur_batch_size] = cur_batch_normals
      batch_weights[0:cur_batch_size] = cur_batch_weights
      # Padding
      PADDING_SIZE = BATCH_SIZE - cur_batch_size
      cur_batch_data, cur_batch_label, cur_batch_normals, cur_batch_weights = get_batch(TRAIN_DATASET, train_idxs, 0, PADDING_SIZE)
      batch_data[cur_batch_size:] = cur_batch_data
      batch_label[cur_batch_size:] = cur_batch_label
      batch_normals[cur_batch_size:] = cur_batch_normals
      batch_weights[cur_batch_size:] = cur_batch_weights

    # Jitter points
    batch_data[:, :, 0:3] = evaluation_utils.jitter_points(batch_data[:, :, 0:3], sigma=0.005, clip=0.01)
    # Perturbate normals
    batch_normals, R = evaluation_utils.jitter_direction_vectors(batch_normals, sigma=1.5, clip=3, return_noise=True)

    # Perturbate tangent vectors
    for data_ind in range(batch_data.shape[0]):
      batch_data[data_ind, :, 3:6] = np.squeeze(np.matmul(R[data_ind], batch_data[data_ind, :, 3:6, np.newaxis]))
      batch_data[data_ind, :, 6:9] = np.squeeze(np.matmul(R[data_ind], batch_data[data_ind, :, 6:9, np.newaxis]))

    # Train
    # Split batch to gpus
    feed_dict = {}
    gpu_batch = BATCH_SIZE / NUM_GPU
    for gpu_idx in range(NUM_GPU):
      start_gpu_batch_idx = gpu_idx * gpu_batch
      end_gpu_batch_idx = (gpu_idx + 1) * gpu_batch
      feed_dict[ops["pointclouds_phs"][gpu_idx]] = batch_data[start_gpu_batch_idx:end_gpu_batch_idx, ...]
      feed_dict[ops["seg_phs"][gpu_idx]] = batch_label[start_gpu_batch_idx:end_gpu_batch_idx, ...]
      feed_dict[ops["is_training_phs"][gpu_idx]] = is_training
      feed_dict[ops["normals_phs"][gpu_idx]] = batch_normals[start_gpu_batch_idx:end_gpu_batch_idx, ...]
      feed_dict[ops["weights_phs"][gpu_idx]] = batch_weights[start_gpu_batch_idx:end_gpu_batch_idx, ...]
    feed_dict[ops['learning_rate_ph']] = learning_rate_np

    if not ROTATE_PRINCIPALS_AROUND_NORMAL:  # Use 2-RoSy field
      seg_pred_val_list = sess.run(ops["seg_pred_list"], feed_dict=feed_dict)
      # Invert principal directions
      batch_data[..., -6:] = -batch_data[..., -6:]
      for gpu_idx in range(NUM_GPU):
        start_gpu_batch_idx = gpu_idx * gpu_batch
        end_gpu_batch_idx = (gpu_idx + 1) * gpu_batch
        feed_dict[ops["pointclouds_phs"][gpu_idx]] = batch_data[start_gpu_batch_idx:end_gpu_batch_idx, ...]
        feed_dict[ops["seg_pred_phs"][gpu_idx]] = seg_pred_val_list[gpu_idx]
    # train_op is for all gpus, and the others are for the last gpu
    _, loss_val, per_instance_seg_loss_val, seg_pred_val_list, pred_seg_res = sess.run([ops["train_op"], ops["loss"],
                                                                                   ops["per_instance_seg_loss"],
                                                                                   ops["seg_pred_list"],
                                                                                   ops["per_instance_seg_pred_res"]],
                                                                                   feed_dict=feed_dict)

    # Save batch cnt
    EPOCH_DICT["cur_train_batch"] = batch_idx + 1
    save_epoch_dict()

    # Calculate accuracy
    seg_gt = batch_label[start_gpu_batch_idx:end_gpu_batch_idx, ...]
    per_instance_part_acc = np.mean(pred_seg_res == seg_gt, axis=1)
    average_part_acc = np.mean(per_instance_part_acc)
    total_loss += loss_val
    total_seg_acc += average_part_acc

    if batch_idx % 10 == 0:

      # Save variables to disk
      saver.save(sess, os.path.join(TRAINED_MODEL_STORAGE_PATH, "model.ckpt"))
      printout(flog, 'Batch: %d --------------------------------' %(batch_idx))
      printout(flog, '          Training Seg Accuracy: %f' % (total_seg_acc * 1.0 / (batch_idx+1)))
      printout(flog, '          Training Loss: %f' % (total_loss * 1.0 / (batch_idx + 1)))

    # Run evaluation metrics
    feed_dict = {
      ops["total_training_loss_ph"]: total_loss * 1.0 / (batch_idx + 1),
      ops["seg_training_acc_ph"]: total_seg_acc * 1.0 / (batch_idx + 1),
      ops['learning_rate_ph']: learning_rate_np
    }

    lr_sum, bn_decay_sum, batch_sum, train_loss_sum, train_seg_acc_sum = sess.run([ops["lr_op"], ops["bn_decay_op"],
                                                                                  ops["batch_op"],
                                                                                  ops["total_train_loss_sum_op"],
                                                                                  ops["seg_train_acc_sum_op"]],
                                                                                  feed_dict=feed_dict)

    # Log metrics
    writer.add_summary(train_loss_sum, batch_idx + epoch * num_batches)
    writer.add_summary(lr_sum, batch_idx + epoch * num_batches)
    writer.add_summary(bn_decay_sum, batch_idx + epoch * num_batches)
    writer.add_summary(train_seg_acc_sum, batch_idx + epoch * num_batches)
    writer.add_summary(batch_sum, batch_idx + epoch * num_batches)

  total_loss = total_loss * 1.0 / num_batches
  total_seg_acc = total_seg_acc * 1.0 / num_batches
  printout(flog, '\tTraining Total Seg Accuracy: %f' % total_seg_acc)
  printout(flog, '\tTraining Total Mean_loss: %f' % total_loss)

  # Lock training
  EPOCH_DICT["train_flag"] = False
  # Reset batch idx
  EPOCH_DICT["cur_train_batch"] = 0
  save_epoch_dict()
##### End train_one_epoch


def eval_one_epoch(sess, epoch, ops, writer):
  global EPOCH_DICT

  is_training = False

  # Init variables for accuracy, chamfer distance and loss
  total_seen = 0; total_seg_acc = 0.0
  total_loss = 0; overall_dcd = 0; exclude_models = 0

  test_idxs = np.arange(0, len(VAL_DATASET))
  # Test on all data: last batch might be smaller than BATCH_SIZE
  test_batch_size = BATCH_SIZE / NUM_GPU
  num_batches = (len(VAL_DATASET) + test_batch_size - 1) / test_batch_size

  # Init batch data
  batch_data = np.zeros((test_batch_size, NUM_POINT, n_features))
  batch_label = np.zeros((test_batch_size, NUM_POINT, 1)).astype(np.int32)
  batch_normals = np.zeros((test_batch_size, NUM_POINT, 3)).astype(np.float32)
  batch_weights = np.zeros((test_batch_size, 1)).astype(np.float32)
  if ABC_DATASET:
    batch_boundary_points = np.zeros((test_batch_size, NUM_POINT, 3)).astype(np.float32)

  # Iterate through all batches
  for batch_idx in range(num_batches):

    start_idx = batch_idx * test_batch_size
    end_idx = min(len(VAL_DATASET), (batch_idx + 1) * test_batch_size)
    cur_batch_size = end_idx - start_idx
    if ABC_DATASET:
      cur_batch_data, cur_batch_label, cur_batch_normals, cur_batch_weights, cur_batch_boundary_points = \
        get_batch_evaluation(VAL_DATASET, test_idxs, start_idx, end_idx)
    else:
      cur_batch_data, cur_batch_label, cur_batch_normals, cur_batch_weights = get_batch(VAL_DATASET, test_idxs, start_idx, end_idx)
    if cur_batch_size == test_batch_size:
      batch_data = cur_batch_data
      batch_label = cur_batch_label
      batch_normals = cur_batch_normals
      batch_weights = cur_batch_weights
      if ABC_DATASET:
        batch_boundary_points = cur_batch_boundary_points
    else:  # Last batch might be smaller than the BATCH_SIZE
      batch_data[0:cur_batch_size] = cur_batch_data
      batch_label[0:cur_batch_size] = cur_batch_label
      batch_normals[0:cur_batch_size] = cur_batch_normals
      batch_weights[0:cur_batch_size] = cur_batch_weights
      if ABC_DATASET:
        batch_boundary_points[0:cur_batch_size] = cur_batch_boundary_points

    # Jitter points
    batch_data[:, :, 0:3] = evaluation_utils.jitter_points(batch_data[:, :, 0:3], sigma=0.005, clip=0.01)
    # Perturbate normals
    batch_normals, R = evaluation_utils.jitter_direction_vectors(batch_normals, sigma=1.5, clip=3, return_noise=True)

    # Perturbate tangent vectors
    for data_ind in range(batch_data.shape[0]):
      batch_data[data_ind, :, 3:6] = np.squeeze(np.matmul(R[data_ind], batch_data[data_ind, :, 3:6, np.newaxis]))
      batch_data[data_ind, :, 6:9] = np.squeeze(np.matmul(R[data_ind], batch_data[data_ind, :, 6:9, np.newaxis]))

    # Inference
    # Run on gpu_(#NUM_GPU-1), since the tensors used for evaluation are defined on gpu_(#NUM_GPU-1)
    feed_dict = {ops["pointclouds_phs"][-1]: batch_data,
                 ops["normals_phs"][-1]: batch_normals,
                 ops["seg_phs"][-1]: batch_label,
                 ops["is_training_phs"][-1]: is_training,
                 ops["weights_phs"][-1]: batch_weights,
                 ops['learning_rate_ph']: learning_rate_np}

    if not ROTATE_PRINCIPALS_AROUND_NORMAL:  # Use 2-RoSy field
      seg_pred_val = sess.run(ops["seg_pred_list"][-1], feed_dict=feed_dict)
      # Invert principal directions
      batch_data[..., -6:] = -batch_data[..., -6:]
      feed_dict[ops["pointclouds_phs"][-1]] = batch_data
      feed_dict[ops["seg_pred_phs"][-1]] = seg_pred_val
    loss_val, per_instance_seg_loss_val, seg_pred_val, pred_seg_res =\
      sess.run([ops["loss"], ops["per_instance_seg_loss"], ops["seg_pred_list"][-1], ops["per_instance_seg_pred_res"]],
               feed_dict=feed_dict)

    # Calculate accuracy
    seg_gt = cur_batch_label
    per_instance_part_acc = np.mean(pred_seg_res[0:cur_batch_size] == seg_gt, axis=1)
    average_part_acc = np.mean(per_instance_part_acc)
    total_seen += 1
    total_loss += loss_val
    total_seg_acc += average_part_acc

    for data_ind in range(cur_batch_size):
      if ABC_DATASET:
        boundary_points = batch_boundary_points[data_ind]
        if len(boundary_points) == 0:
          # Ground truth segmentation -> exclude from evaluation
          exclude_models += 1
        else:
          seg_2_boundaries = np.unique(pred_seg_res[data_ind])
          if len(seg_2_boundaries) == 1:
            # Predicted segmentation -> penalize for chamfer distance
            overall_dcd += evaluation_utils.pc_bounding_box_diagonal(batch_data[data_ind, :, 0:3])
          else:
            # Evaluate inferred boundaries wrt. chamfer distance
            segp = np.squeeze(pred_seg_res[data_ind])
            dcd = evaluation_utils.chamfer_distance(segmentation1_points=batch_data[data_ind, segp==1, 0:3],
                                                    segmentation2_points=boundary_points,
                                                    points=batch_data[data_ind, :, 0:3])
            overall_dcd += dcd
      else:
        seg_1_boundaries = np.unique(batch_label[data_ind])
        if len(seg_1_boundaries) == 1:
          # Ground truth segmentation -> exclude from evaluation
          exclude_models += 1
        else:
          seg_2_boundaries = np.unique(pred_seg_res[data_ind])
          if len(seg_2_boundaries) == 1:
            # Predicted segmentation -> penalize for chamfer distance
            overall_dcd += evaluation_utils.pc_bounding_box_diagonal(batch_data[data_ind, :, 0:3])
          else:
            # Evaluate inferred boundaries wrt. chamfer distance
            dcd = evaluation_utils.chamfer_distance(segmentation1=np.squeeze(batch_label[data_ind]),
                                                    segmentation2=np.squeeze(pred_seg_res[data_ind]),
                                                    points=batch_data[data_ind, :, 0:3])
            overall_dcd += dcd

  # Run evaluation metrics`
  total_loss = total_loss * 1.0 / total_seen
  total_seg_acc = total_seg_acc * 1.0 / total_seen
  try:
    total_dcd = overall_dcd / float(len(VAL_DATASET) - exclude_models)
  except ZeroDivisionError:
    total_dcd = 1e5
  feed_dict = {ops["total_testing_loss_ph"]: total_loss,
               ops["seg_testing_acc_ph"]: total_seg_acc,
               ops["total_testing_dcd_ph"]: total_dcd}

  test_loss_sum, test_seg_acc_sum, test_dcd_sum = sess.run([ops["total_test_loss_sum_op"], ops["seg_test_acc_sum_op"],
                                                            ops["total_test_dcd_sum_op"]],
                                                            feed_dict=feed_dict)
  # Log metrics
  train_num_batches = (len(TRAIN_DATASET) + BATCH_SIZE - 1) / BATCH_SIZE
  writer.add_summary(test_loss_sum, (epoch+1) * train_num_batches)
  writer.add_summary(test_seg_acc_sum, (epoch+1) * train_num_batches)
  writer.add_summary(test_dcd_sum, (epoch + 1) * train_num_batches)

  printout(flog, '\tTesting Seg Accuracy: %f' % total_seg_acc)
  printout(flog, '\tTesting DCD: %f' % total_dcd)
  printout(flog, '\tTesting Total Loss: %f' % total_loss)

  # Reduce learning rate based on val metric
  patience = 5
  if reduce_lr_on_plateau(cur_metric=total_loss, patience=patience, factor=0.5, min_lr=LEARNING_RATE_CLIP,
                          mode='min'):
    printout(flog, 'learning was reduced to %f, after %d epochs' % (learning_rate_np, patience))

  return total_seg_acc, total_dcd, total_loss
##### End eval_one_epoch


if __name__=='__main__':
  train()