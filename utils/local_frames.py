import tensorflow as tf


def construct_local_frames(points, normals, tangent_vec1, tangent_vec2, return_R=False, return_t=False):
    """Construct local frame for each point
        Input:
                points: (batch_size, n_points, 3) TF tensor
                normals: (batch_size, n_points, 3) TF tensor
                tangent_vec1: (batch_size, n_points, 3) TF tensor
                tangent_vec2: (batch_size, n_points, 3) TF tensor
         Return:
                M: (batch_size, n_points, 3, 4) TF tensor, local frame for each point
    """

    # Build local frame for each point
    R = tf.concat([tangent_vec1, tangent_vec2, normals], axis=-1)
    R = tf.reshape(R, [points.shape[0].value, points.shape[1].value, 3, 3])
    points = tf.expand_dims(points, axis=3)
    t = -tf.matmul(R, points)
    M = tf.concat([R, t], axis=3)

    transformations = (M,)
    if return_R:
        transformations += (R, )
    if return_t:
        transformations += (t,)

    return transformations


def transform_neighbors(neighbors, M):
    """ Transform all neighboring points of a centroid through the transformation M
        Input:
                neighbors: (batch_size, n_centroids, n_neighbors, 3) TF tensor
                M: (batch_size, n_centroids, 3, 4) TF tensor, local frame for each centroid
        Return:
                transformed_neighbors = (batch_size, n_centroids, n_neighbors, 3)
    """

    # Sanity check
    B = neighbors.shape[0]; C = neighbors.shape[1];  N = neighbors.shape[2]; D = neighbors.shape[3]
    assert (M.shape[0] == B); assert (M.shape[1] == C); assert (M.shape[2] == D);
    assert (M.shape[3] == 4);

    # Project all neighbors
    ones = tf.ones((neighbors.shape[0].value, neighbors.shape[1].value, neighbors.shape[2].value, 1), dtype=tf.float32)
    h_neighbors = tf.concat([neighbors, ones], axis=3)
    h_neighbors = tf.transpose(h_neighbors, [0,1,3,2])
    projected_neighbors = tf.matmul(M, h_neighbors)
    projected_neighbors = tf.transpose(projected_neighbors, [0,1,3,2])

    return projected_neighbors


def rotate_vectors(vectors, R):
    """ Transform vectors using rotation matrix R
        Input:
                vectors: (batch_size, n_centroids, n_neighbors, 3) TF tensor
                R: (batch_size, n_points, 3, 3) TF tensor, rotation matrix
        Return:
                rotated_vectors = (batch_size, n_centroids, n_neighbors, 3)
    """

    # Rotate vectors
    rotated_vectors = tf.matmul(R, tf.transpose(vectors, [0,1,3,2]))

    return tf.transpose(rotated_vectors, [0,1,3,2])


def project_neighbors_to_local_frame(centroids, neighbors, normals, tangent_vec1, tangent_vec2):
    """Project each neighbor of the centroid to its local frame
    Input:
            centroids: (batch_size, n_centroids, 3) TF tensor
            neighbors: (batch_size, n_centroids, n_neighbors, 3) TF tensor
            normals: (batch_size, n_centroids, 3) TF tensor
            tangent_vec1: (batch_size, n_centroids, 3) TF tensor
            tangent_vec2: (batch_size, n_centroids, 3) TF tensor
    Return:
            projected_neighbors: (batch_size, n_centroids, n_neighbors, 3) TF tensor, local coordinates of each
            neighbor sample w.r.t. the local frame of their centroid
    """

    # Sanity check
    B = centroids.shape[0]; C = centroids.shape[1]
    assert (neighbors.shape[0] == B); assert (neighbors.shape[1] == C)

    # Get local frame for each centroid
    M = construct_local_frames(points=centroids, normals=normals, tangent_vec1=tangent_vec1,
                               tangent_vec2=tangent_vec2)[0]

    # Project all neighbors
    projected_neighbors = transform_neighbors(neighbors=neighbors, M=M)

    return projected_neighbors