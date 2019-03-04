import cv2
import numpy as np
import tensorflow as tf


def to_intrinsics(f, cu, cv):
    return tf.constant([[float(f), 0., float(cu)], [0., float(f), float(cv)], [0., 0., 1.]])


def repeat(mat, num, axis=0):
    return tf.stack([mat for i in range(num)], axis=axis)


def repeat2(mat, num, axis=0):
    """repeat a row with broadcasting is that faster than tf.stack?"""
    num_dims = len(mat.shape.as_list())
    if num_dims < axis:
        raise Exception("In repeat2: dimension to expand smaller than matrix")
    shape = [1 for i in range(num_dims)]
    shape[axis] = num
    return tf.ones(shape) * mat


def get_rotation(angle, axis=0):
    """

    :param angle: angles of shape(batch_size,1)
    :param axis: axis around which we should rotate
    :return: rotation matrics of shape (batch_size,3,3)
    """
    zeros = tf.zeros_like(angle)
    ones = tf.ones_like(angle)
    if axis == 0:
        stacked = tf.stack(
            [[ones, zeros, zeros], [zeros, tf.cos(angle), -tf.sin(angle)], [zeros, tf.sin(angle), tf.cos(angle)]])
    elif axis == 1:
        stacked = tf.stack(
            [[tf.cos(angle), zeros, tf.sin(angle)], [zeros, ones, zeros], [-tf.sin(angle), zeros, tf.cos(angle)]])
        return tf.transpose(stacked, (2, 0, 1))
    elif axis == 2:
        stacked = tf.stack(
            [[tf.cos(angle), -tf.sin(angle), zeros], [tf.sin(angle), tf.cos(angle), zeros], [zeros, zeros, ones]])
    else:
        raise Exception("Wrong axis defined.")

    # swp dimensions so that batch_size ist first dimension
    return tf.transpose(stacked, (2, 0, 1))


def get_cross_mat(t):
    """

    :param t: translation with shape(batch_size,3,1)
    :return: cross prod matrices with shape(batch_size, 3, 3)
    """
    zeros = tf.zeros_like(t[:, 0])
    cross_t = tf.stack(
        [[zeros, -t[:, 2], t[:, 1]], [t[:, 2], zeros, -t[:, 0]], [-t[:, 1], t[:, 0], zeros]])
    cross_t = tf.squeeze(cross_t, axis=3)
    cross_t = tf.transpose(cross_t, (2, 0, 1))
    return cross_t


def get_translation_rotation(angles):
    """transform from rpy, trans_yaw, trans_pitch to translation and rotation matrices"""
    batch_size, five = angles.shape.as_list()

    assert (five == 5)
    t = tf.constant([0., 0., 1.], dtype=angles.dtype)
    # Hacky emulation of np.repeat
    t = repeat(t, batch_size)
    t = tf.expand_dims(t, axis=2)
    # Now t should have shape(batch_size,3,1)
    # Apply translation yaw (camera coordinates!)
    t = tf.matmul(get_rotation(angles[:, 3], axis=1), t)
    # Apply translation pitch (camera coordinates!)
    t = tf.matmul(get_rotation(angles[:, 4], axis=0), t)

    # Rotation definition as in here http://planning.cs.uiuc.edu/node102.html
    roll = get_rotation(angles[:, 0], axis=2)
    pitch = get_rotation(angles[:, 1], axis=0)
    yaw = get_rotation(angles[:, 2], axis=1)
    rotation = tf.matmul(yaw, tf.matmul(pitch, roll))

    return t, rotation


def get_fundamental_matrix(angles, intrin):
    """
    :param angles: shape(batch_size,5) roll, pitch, yaw, trans_yaw, trans_pitch
    :param intrin: matrix with intrinsics as constant with shape (batch_size,3,3)
    :return: fundamental matrices, shape (batch_size,3,3)
    """
    t, rotation = get_translation_rotation(angles)

    # Get cross matrix
    cross_t = get_cross_mat(t)

    # Calculate Essential Matrix as in multiple view geometry p.257
    # should have shape (batch_size,3,3)
    E = tf.matmul(cross_t, rotation)

    # Calculate Fundamtenal matrix
    intrin_inv = tf.matrix_inverse(intrin)
    F = tf.matmul(intrin_inv, tf.matmul(E, intrin_inv), transpose_a=True)
    return F


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")


def antipad(tensor, num=1):
    """
    Performs a crop. "padding" for a deconvolutional layer (conv2d tranpose) removes
    padding from the output rather than adding it to the input.
    """
    batch, h, w, c = tensor.shape.as_list()
    return tf.slice(tensor, begin=[0, num, num, 0], size=[batch, h - 2 * num, w - 2 * num, c])


def reform_hartley(flow):
    """Reforms the flow in image form to the matrix defined by hartley for the 8-point algorithm.
        See 'Multiple View Geometry' by Hartley and Zissermann, p.279"""

    # A=[x'*x, x'*y,x',y'*x,y'*y,y',x,y,1], with (x',y') current point, (x,y) last point
    # last points are all image coordinates
    # current points are last points + flow
    # so with (x',y') = (x+dx,y+dy) follows A = [x^2+x*dx, x*y+dx*y, dx+x,y*x+x*dy,y^2+y*dy,y+dy,x,y,1]

    batch_num, height, width, num_flow_layers = flow.shape.as_list()

    assert (num_flow_layers == 2)

    # sess = tf.Session()

    l = []
    for _x in range(10, width, int(width / 100)):
        for _y in range(10, height, int(height / 100)):
            dx = flow[:, _y, _x, 0]
            # with sess.as_default():
            #    print(dx.eval())
            dy = flow[:, _y, _x, 1]
            x = float(_x)
            y = float(_y)

            ones = tf.ones_like(dx)
            l.append([tf.stack(
                [x ** 2 + tf.scalar_mul(x, dx), x * y + tf.scalar_mul(y, dx), dx + x, y * x + tf.scalar_mul(x, dy),
                 y ** 2 + tf.scalar_mul(y, dy), y + dy, ones * x, ones * y, ones])])

    # A should have size (num_batches, height*width, 9)
    A = tf.stack(l, axis=1)
    A = tf.squeeze(A, axis=3)
    return A


# def epipolar_errors(predict_fundamental_matrix, flow):
#     """
#     return: a tensor with shape (num_batchs, height*width) with the epipolar errors of the flow given the
#     fundamental matrix prediction with shape (num_batches, 9, 1).
#     input:
#     - Prediction of fundamental matrix in form (f_11,f_12,f_13,f_21,f_22,f_23,f_31,f_32,f_33)
#     - Estimated flow image (shape=(num_batches, height, width, 2))
#     """
#
#     batch_size, height, width = predict_fundamental_matrix.shape.as_list()
#     # Translate in matrix hartley style (A*f=err), so I don't have to reshape and save 1 multiplication.
#     A = reform_hartley(flow)
#     if not (height == 9 and width == 1):
#         raise Exception("Wrong in put dimensions height={} width={}".format(height, width))
#
#     # predict_fundamental_matrix = predict_fundamental_matrix_in
#     # elif height == 3 and width == 3:
#     #     predict_fundamental_matrix = tf.reshape(predict_fundamental_matrix_in, (batch_size, 9, 1))
#     # else:
#     error_vec = tf.matmul(A, predict_fundamental_matrix)  # check dimensions.
#     return error_vec

def get_image_coordinates_u(shape):
    batch_size, height, width = shape
    # create matrix with column numbers in each row
    u = [[[i for i in range(width)] for j in range(height)] for bs in range(batch_size)]

    return tf.constant(u, dtype=tf.float32)


def get_image_coordinates_v(shape):
    batch_size, height, width = shape
    # create matrix with column numbers in each row
    v = [[[j for i in range(width)] for j in range(height)] for bs in range(batch_size)]

    return tf.constant(v, dtype=tf.float32)


def get_image_coordinates_as_points(shape):
    """
    Get image point coordinates as flow vector in homogenous coordinates.
    Used for fundamental matrix evaluation from dense flow.
    :param shape: (batch_size, flow_height, flow_width)
    :return: pixel coordinates in homogenous coordinates as tensor of shape (batch_size, flow_height*flow_width, 3)
    """
    u = get_image_coordinates_u(shape)
    u = tf.expand_dims(u, axis=3)
    v = get_image_coordinates_v(shape)
    v = tf.expand_dims(v, axis=3)

    uv1 = tf.stack((u, v, tf.ones_like(u)), axis=3)
    uv1 = tf.cast(uv1, dtype=tf.float32)

    uv1_vec = tf.reshape(uv1, (shape[0], shape[1] * shape[2], 3))

    return uv1_vec


def get_correspondences(flow):
    batch_size, flow_h, flow_w, two = flow.shape.as_list()
    assert (two == 2)

    # Get image point coordinates as flow vector in homogeneous coordinates.
    old_points = get_image_coordinates_as_points((batch_size, flow_h, flow_w))
    old_points = tf.transpose(old_points, (0, 2, 1))

    # Add calculated flow to images coordinates to get new flow.
    flow_vec = tf.reshape(flow, (batch_size, flow_h * flow_w, two))
    flow_vec = tf.transpose(flow_vec, (0, 2, 1))
    flow_vec = tf.concat((flow_vec, tf.zeros((batch_size, 1, flow_h * flow_w))), axis=1)
    new_points = old_points + flow_vec

    return old_points, new_points


def calc_essential_matrix_5point(flow, intrinsics):
    f = intrinsics[0][0]
    pp = (intrinsics[0][2], intrinsics[1][2])
    old_points, new_points = get_correspondences(flow)
    E = cv2.findEssentialMat(old_points, new_points, focal=f, pp=pp, method="MEDS")

    return E


def normalize_feature_points(old_points, new_points):
    bs, _, l = old_points.shape.as_list()

    # shift origins to centroids
    cpu = tf.reduce_sum(old_points[:, 0, :], axis=1, keepdims=True)
    cpv = tf.reduce_sum(old_points[:, 1, :], axis=1, keepdims=True)
    ccu = tf.reduce_sum(new_points[:, 0, :], axis=1, keepdims=True)
    ccv = tf.reduce_sum(new_points[:, 1, :], axis=1, keepdims=True)

    cpu /= l
    cpv /= l
    ccu /= l
    ccv /= l

    old_points = old_points - tf.stack((cpu, cpv, tf.zeros_like(cpu)), axis=1)
    new_points = new_points - tf.stack((ccu, ccv, tf.zeros_like(ccu)), axis=1)

    # scale features such that mean distance from origin is sqrt(2)
    sp = tf.reduce_sum(tf.sqrt(tf.square(old_points[:, 0, :]) + tf.square(old_points[:, 1, :])), axis=1, keepdims=True)
    sc = tf.reduce_sum(tf.sqrt(tf.square(new_points[:, 0, :]) + tf.square(new_points[:, 1, :])), axis=1, keepdims=True)

    # if (fabs(sp)<1e-10 || fabs(sc)<1e-10)
    #  return false;
    sp = tf.sqrt(2.0) * l / sp
    sc = tf.sqrt(2.0) * l / sc

    old_points = tf.multiply(old_points, tf.stack((sp, sp, tf.ones_like(sp)), axis=1))
    new_points = tf.multiply(new_points, tf.stack((sc, sc, tf.ones_like(sc)), axis=1))
    # compute corresponding transformation matrices
    Tp = tf.squeeze(tf.concat([tf.expand_dims(tf.stack([sp, tf.zeros_like(sp), -sp * cpu], axis=1), axis=1),
                               tf.expand_dims(tf.stack([tf.zeros_like(sp), sp, -sp * cpv], axis=1), axis=1),
                               tf.expand_dims(
                                   tf.stack([tf.zeros_like(sp), tf.zeros_like(sp), tf.ones_like(sp)], axis=1), axis=1)],
                              axis=1), axis=3)
    Tc = tf.squeeze(tf.concat([tf.expand_dims(tf.stack([sc, tf.zeros_like(sc), -sc * ccu], axis=1), axis=1),
                               tf.expand_dims(tf.stack([tf.zeros_like(sc), sc, -sc * ccv], axis=1), axis=1),
                               tf.expand_dims(
                                   tf.stack([tf.zeros_like(sp), tf.zeros_like(sp), tf.ones_like(sp)], axis=1), axis=1)],
                              axis=1), axis=3)

    old_points = tf.cast(old_points, dtype=tf.float32)
    new_points = tf.cast(new_points, dtype=tf.float32)
    Tp = tf.cast(Tp, dtype=tf.float32)
    Tc = tf.cast(Tc, dtype=tf.float32)

    return old_points, new_points, Tp, Tc


def epipolar_squared_errors_to_prob(error_vec):
    error_vec = tf.reduce_max(error_vec, axis=1, keepdims=True) - tf.sqrt(
        tf.clip_by_value(error_vec, clip_value_min=1e-100, clip_value_max=1e100))
    norm = tf.clip_by_value(tf.reduce_sum(error_vec, axis=1, keepdims=True), clip_value_min=1e-100,
                            clip_value_max=1e100)
    error_vec = tf.divide(error_vec, norm)
    return error_vec


def calc_fundamental_matrix_8point(flow, inlier_prob=None, number_iterations=10):
    def fundamental_matrix(old_points, new_points, weights):
        bs = old_points.shape.as_list()[0]
        l = old_points.shape.as_list()[2]

        # create constraint matrix A
        A = tf.stack((new_points[:, 0, :] * old_points[:, 0, :], new_points[:, 0, :] * old_points[:, 1, :],
                      new_points[:, 0, :], new_points[:, 1, :] * old_points[:, 0, :],
                      new_points[:, 1, :] * old_points[:, 1, :], new_points[:, 1, :], old_points[:, 0, :],
                      old_points[:, 1, :], tf.ones_like(old_points[:, 0, :])), axis=2)

        A_weighted = tf.multiply(A, weights)
        # compute singular value decomposition of A
        singular_vals, _, V = tf.linalg.svd(A_weighted)

        # extract fundamental matrix from the column of V corresponding to the smallest singular value
        assert (V.shape.as_list()[1] == 9)
        F = tf.reshape(V[:, :, -1], (-1, 3, 3))

        # enforce rank 2
        singular_vals, U, V = tf.linalg.svd(F)
        singular_vals = tf.concat((singular_vals[:, :-1], tf.zeros((singular_vals.shape.as_list()[0], 1))), axis=1)
        F = tf.matmul(tf.matmul(U, tf.matrix_diag(singular_vals)), tf.matrix_transpose(V))
        return F

    old_points, new_points = get_correspondences(flow)

    old_points, new_points, Tp, Tc = normalize_feature_points(old_points, new_points)

    F = None

    if inlier_prob is None:
        weights = tf.expand_dims(tf.ones_like(old_points)[:, 0, :], axis=2)
    else:
        weights = tf.reshape(inlier_prob,
                             (flow.shape.as_list()[0], flow.shape.as_list()[1] * flow.shape.as_list()[2], 1))

    for i in range(number_iterations):
        F = fundamental_matrix(old_points, new_points, weights)
        # denormalize
        F = tf.matmul(tf.matmul(tf.matrix_transpose(Tc), F), Tp)

        # update weights
        weights = epipolar_squared_errors_to_prob(epipolar_errors_squared(F, flow))
        weights = tf.expand_dims(weights, axis=2)

    return F


def threshold(x, thres):
    cond = tf.less(x, tf.scalar_mul(thres, tf.ones(tf.shape(x))))
    out = tf.where(cond, tf.zeros(tf.shape(x)), tf.ones(tf.shape(x)))
    return out


def get_mask_fundamental_mat(flow, inlier_thres=1e-8, inlier_probs=None):
    fundamental_mat = calc_fundamental_matrix_8point(flow, inlier_probs)


def get_mask_fundamental_mat(flow, inlier_thres=None, inlier_probs=None):
    fundamental_mat = calc_fundamental_matrix_8point(flow, inlier_probs)

    if inlier_thres is not None:
        new_inlier_probs = tf.reshape(epipolar_errors_squared(fundamental_mat, flow), flow.shape.as_list()[:3])
        new_inlier_probs = 1. - threshold(tf.sqrt(new_inlier_probs), inlier_thres)
    else:
        new_inlier_probs = tf.reshape(epipolar_squared_errors_to_prob(epipolar_errors_squared(fundamental_mat, flow)),
                                      flow.shape.as_list()[:3])
        new_inlier_probs = new_inlier_probs - tf.reduce_min(new_inlier_probs)
        new_inlier_probs = tf.div_no_nan(new_inlier_probs, tf.reduce_max(new_inlier_probs))

    return tf.stack((1. - new_inlier_probs, new_inlier_probs), axis=3)


def epipolar_errors_squared(predict_fundamental_matrix_in, flow, mask_weights=None, *, normalize=True, debug=False):
    """
    return: a tensor with shape (num_batchs, height*width) with the squared epipolar errors of the flow given the
    fundamental matrix prediction with shape (num_batches, 9, 1).
    input:
    - Prediction of fundamental matrix in form (f_11,f_12,f_13,f_21,f_22,f_23,f_31,f_32,f_33)
    - Estimated flow image (shape=(num_batches, height, width, 2))
    """

    batch_size, height, width = predict_fundamental_matrix_in.shape.as_list()

    if height == 9 and width == 1:
        pred_fun = tf.reshape(predict_fundamental_matrix_in, (batch_size, 3, 3))
    elif height == 3 and width == 3:
        pred_fun = predict_fundamental_matrix_in
    else:
        raise Exception("Invalid number of dimensions.")

    old_points, new_points = get_correspondences(flow)

    # Calculate epipolar error.
    Fx = tf.matmul(pred_fun, old_points)  # needed for normalization
    # This is like tf.matrix_diag_part(tf.matmul(new_points, tmp, transpose_a=True)) but with less memory usage.
    # error_vec = tf.einsum('aij,aij->aj', new_points, Fx)  # should be the same
    error_vec = tf.reduce_sum(tf.multiply(new_points, Fx), axis=1)

    # Square the error
    error_vec = tf.multiply(error_vec, error_vec)

    if normalize:
        # Normalize the metric using the sampson distance
        # , which is the first order approximation to the geometric distance (i.e. reprojection error)
        # see Multiple View geometry p.287
        Ftxp = tf.matmul(tf.matrix_transpose(pred_fun), new_points)  # this is F^Tx' form hartley
        Ftxp2 = tf.multiply(Ftxp, Ftxp)  # shape (bs, 3, num_pixels)

        Fx2 = tf.multiply(Fx, Fx)  # shape (bs, 3, num_pixels)

        norm_fact = Fx2[:, 0, :] + Fx2[:, 1, :] + Ftxp2[:, 0, :] + Ftxp2[:, 1, :]  # shape (bs, num_pixels)

        # don't divide by zero
        norm_fact = tf.clip_by_value(norm_fact, clip_value_min=1e-15, clip_value_max=1e30)
        error_vec = tf.divide(error_vec, norm_fact)

    if mask_weights is not None:
        _, flow_h, flow_w, _ = flow.shape.as_list()
        # Do weighting with mask.
        # Get weights as vector with shape (batch_size, height*width)
        assert (len(mask_weights.shape.as_list()) == 3)
        assert (mask_weights.shape.as_list()[1] == flow_h and mask_weights.shape.as_list()[2] == flow_w)
        weights = tf.reshape(mask_weights, (-1, flow_h * flow_w))
        error_vec = tf.multiply(error_vec, weights)

    if debug:
        add_to_debug_output('new_points', new_points)
        add_to_debug_output('fundamental_matrix', pred_fun)
        add_to_debug_output('error', error_vec)

    return error_vec


def get_reference_explain_mask(mask_shape, downscaling=0):
    batch_size, height, width, _ = mask_shape
    tmp = np.array([0, 1])
    ref_exp_mask = np.tile(tmp,
                           (batch_size,
                            int(height / (2 ** downscaling)),
                            int(width / (2 ** downscaling)),
                            1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask


def get_inlier_prob_from_mask_logits(cur_exp_logits, normalize=False):
    cur_exp = tf.nn.softmax(cur_exp_logits)
    inlier_probs = cur_exp[:, :, :, 1]  # inlier prob

    inlier_probs = tf.expand_dims(inlier_probs, axis=3)

    # Normalize probabilities from 1 to zero.
    if normalize:
        inlier_probs = inlier_probs - tf.reduce_min(inlier_probs)
        inlier_probs = tf.div_no_nan(inlier_probs, tf.reduce_max(inlier_probs))

    return inlier_probs


def print_nan_tf(t, msg=""):
    cond = tf.reduce_any(tf.is_nan(t))
    return tf.Print(t, [msg, cond])


def summarized_placeholder(name, prefix=None, key=tf.GraphKeys.SUMMARIES):
    prefix = '' if not prefix else prefix + '/'
    p = tf.placeholder(tf.float32, name=name)
    tf.summary.scalar(prefix + name, p, collections=[key])
    return p


def resize_area(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_area(tensor, [h, w]))


def resize_bilinear(tensor, like):
    _, h, w, _ = tf.unstack(tf.shape(like))
    return tf.stop_gradient(tf.image.resize_bilinear(tensor, [h, w]))


def add_to_debug_output(name, tensor):
    name = 'debug/' + name
    tf.add_to_collection('debug_tensors', tf.identity(tensor, name=name))


def add_to_output(name, tensor):
    tf.add_to_collection('tracked_tensors', tf.identity(tensor, name=name))
