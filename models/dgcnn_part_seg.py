import os
import sys
import tensorflow as tf
import numpy as np

# Import custom packages
BASE_DIR = os.path.abspath(os.pardir)
sys.path.append(BASE_DIR)
from utils import tf_util
from transform_nets import global_spatial_transformer


def get_model(point_cloud, is_training, normals, use_local_frame=True, add_normals=False, bn=True, bn_decay=None,
			  use_xavier=True, align_pointclouds=False, drop_prob=0.5, n_classes=1, k=20):
	""" Part segmentation DGCNN, input is BxNxnFeatures, output BxnClasses """

	batch_size = point_cloud.get_shape()[0].value
	num_point = point_cloud.get_shape()[1].value

	# Input xyz coordinates
	input_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])

	# Add tangent vectors and normals for local frames calculation
	local_frame_data = None
	if use_local_frame:
		tangent_vec1 = tf.slice(point_cloud, [0, 0, 3], [-1, -1, 3], name="tangent_vec1")
		tangent_vec2 = tf.slice(point_cloud, [0, 0, 6], [-1, -1, 3], name="tangent_vec2")
		local_frame_data = (tangent_vec1, tangent_vec2, normals)

	# Point clouds global alignment
	if align_pointclouds:
		# Calculate pairwise distance on global coordinates and find k-nn's for each point
		adj = tf_util.pairwise_distance(input_xyz)
		nn_idx = tf_util.knn(adj, k=k)
		input_xyz = tf.expand_dims(input_xyz, -1)
		edge_feature = tf_util.get_edge_feature(input_xyz, nn_idx=nn_idx, k=k)

		with tf.variable_scope('transform_net_global') as sc:
			global_transform = global_spatial_transformer(point_cloud=edge_feature, is_training=is_training, bn=bn,
														  bn_decay=bn_decay, is_dist=True)
		input_xyz = tf.matmul(tf.squeeze(input_xyz, axis=-1), global_transform)

	if add_normals:
		if input_xyz.shape.ndims == 4:
			input_xyz = tf.squeeze(input_xyz, axis=-1)
		input_xyz = tf.concat([input_xyz, normals], axis=-1)

	input_image = tf.expand_dims(input_xyz, -1)
	adj = tf_util.pairwise_distance(input_xyz)
	nn_idx = tf_util.knn(adj, k=k)
	edge_feature = tf_util.get_edge_feature(input_image, nn_idx=nn_idx, k=k, use_local_frame=use_local_frame,
											local_frame_data=local_frame_data, add_normals=add_normals)

	# EdgeConv layer 1 {64, 64}
	out1 = tf_util.conv2d(edge_feature, 64, [1, 1], padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
						  scope='adj_conv1', bn_decay=bn_decay, is_dist=True, use_xavier=use_xavier)

	out2 = tf_util.conv2d(out1, 64, [1, 1], padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
						  scope='adj_conv2', bn_decay=bn_decay, is_dist=True, use_xavier=use_xavier)

	net_1 = tf.reduce_max(out2, axis=-2, keep_dims=True)

	# EdgeConv layer 2 {64, 64}
	adj = tf_util.pairwise_distance(net_1)
	nn_idx = tf_util.knn(adj, k=k)
	edge_feature = tf_util.get_edge_feature(net_1, nn_idx=nn_idx, k=k)

	out3 = tf_util.conv2d(edge_feature, 64, [1, 1], padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
						  scope='adj_conv3', bn_decay=bn_decay, is_dist=True, use_xavier=use_xavier)

	out4 = tf_util.conv2d(out3, 64, [1, 1], padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
						  scope='adj_conv4', bn_decay=bn_decay, is_dist=True, use_xavier=use_xavier)

	net_2 = tf.reduce_max(out4, axis=-2, keep_dims=True)

	# EdgeConv layer 3 {64}
	adj = tf_util.pairwise_distance(net_2)
	nn_idx = tf_util.knn(adj, k=k)
	edge_feature = tf_util.get_edge_feature(net_2, nn_idx=nn_idx, k=k)

	out5 = tf_util.conv2d(edge_feature, 64, [1, 1], padding='VALID', stride=[1, 1], bn=bn, is_training=is_training,
						  scope='adj_conv5', bn_decay=bn_decay, is_dist=True, use_xavier=use_xavier)

	net_3 = tf.reduce_max(out5, axis=-2, keep_dims=True)

	# [EdgeConv1, EdgeConv2, EdgeConv3] -> MLP {64}
	out7 = tf_util.conv2d(tf.concat([net_1, net_2, net_3], axis=-1), 1024, [1, 1], padding='VALID', stride=[1, 1],
						  bn=bn, is_training=is_training, scope='adj_conv7', bn_decay=bn_decay, is_dist=True,
						  use_xavier=use_xavier)

	out_max = tf_util.max_pool2d(out7, [num_point, 1], padding='VALID', scope='maxpool')
	expand = tf.tile(out_max, [1, num_point, 1, 1])

	# Concat [global_feature, EdgeConv1, EdgeConv2, EdgeConv3]
	concat = tf.concat(axis=3, values=[expand, net_1, net_2, net_3])

	# FC layer - MLP{256, 256, 128, n_classes}
	net2 = tf_util.conv2d(concat, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay, bn=bn,
						  is_training=is_training, scope='seg/conv1', is_dist=True, use_xavier=use_xavier)
	net2 = tf_util.dropout(net2, keep_prob=1-drop_prob, is_training=is_training, scope='seg/dp1')
	net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay, bn=bn,
						  is_training=is_training, scope='seg/conv2', is_dist=True, use_xavier=use_xavier)
	net2 = tf_util.dropout(net2, keep_prob=1-drop_prob, is_training=is_training, scope='seg/dp2')
	net2 = tf_util.conv2d(net2, 128, [1, 1], padding='VALID', stride=[1, 1], bn_decay=bn_decay, bn=bn,
						  is_training=is_training, scope='seg/conv3', is_dist=True, use_xavier=use_xavier)
	net2 = tf_util.conv2d(net2, n_classes, [1, 1], padding='VALID', stride=[1, 1], activation_fn=None, bn=False,
						  scope='seg/conv4', is_dist=False, use_xavier=use_xavier)

	net2 = tf.reshape(net2, [batch_size, num_point, n_classes])

	return net2


def get_loss(**kwargs):
	mandatory_args = ["seg_pred", "seg"]
	for arg in mandatory_args:
		if arg not in kwargs.keys():
			print("ERROR: {arg:s} not specified for calling get_loss()" .format(arg=arg))
			exit(-1)

	if "weights" in kwargs.keys():
		# Call sigmoid + weighted cross-entropy
		return _get_loss_with_weights(seg_pred=kwargs["seg_pred"], seg=kwargs["seg"], weights=kwargs["weights"])

	# Call softmax + cross-entropy
	return _get_loss(seg_pred=kwargs["seg_pred"], seg=kwargs["seg"])


def _get_loss(seg_pred, seg):
	per_instance_seg_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=seg_pred, labels=seg),
										   axis=1)
	seg_loss = tf.reduce_mean(per_instance_seg_loss)
	per_instance_seg_pred_res = tf.argmax(seg_pred, 2)

	return seg_loss, per_instance_seg_loss, per_instance_seg_pred_res


def _get_loss_with_weights(seg_pred, seg, weights):
	per_instance_seg_loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=tf.squeeze(seg),
																					logits=tf.squeeze(seg_pred),
																					pos_weight=weights), axis=1)
	seg_loss = tf.reduce_mean(per_instance_seg_loss)

	return seg_loss, per_instance_seg_loss