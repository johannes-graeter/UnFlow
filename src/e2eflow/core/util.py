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


def get_fundamental_matrix(angles, intrin):
    """
    :param angles: shape(batch_size,5) roll, pitch, yaw, trans_yaw, trans_pitch
    :param intrin: matrix with intrinsics as constant with shape (3,3)
    :return: fundamental matrices, shape (batch_size,3,3)
    """
    batch_size, five = angles.shape.as_list()

    assert (five == 5)
    t = tf.constant([0., 0., 1.])
    # Hacky emulation of np.repeat
    t = repeat(t, batch_size)
    t = tf.expand_dims(t, axis=2)
    # Now t should have shape(batch_size,3,1)
    # Apply yaw (camera coordinates!)
    t = tf.matmul(get_rotation(angles[:, 3], axis=1), t)
    # Apply pitch (camera coordinates!)
    t = tf.matmul(get_rotation(angles[:, 4], axis=0), t)
    # Get cross matrix
    cross_t = get_cross_mat(t)

    # Rotation definition as in here http://planning.cs.uiuc.edu/node102.html
    roll = get_rotation(angles[:, 0], axis=2)
    pitch = get_rotation(angles[:, 1], axis=0)
    yaw = get_rotation(angles[:, 2], axis=1)
    rotation = tf.matmul(yaw, tf.matmul(pitch, roll))

    # Calculate Essential Matrix as in multiple view geometry p.257
    # should have hape (batch_size,3,3)
    E = tf.matmul(cross_t, rotation)

    # E = tf.Print(E, ["E", E[:, :, :]])
    # intrin = tf.Print(intrin, ["Intrinsics", *[intrin[i, j] for i in range(3) for j in range(3)]])

    # Calculate Fundamtenal matrix
    intrin_inv = tf.matrix_inverse(intrin)

    intrin_inv_tile = repeat(intrin_inv, batch_size)
    F = tf.matmul(intrin_inv_tile, tf.matmul(E, intrin_inv_tile), transpose_a=True)
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
#     print("before reform")
#     A = reform_hartley(flow)
#     print("after reform")
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
    uv1_vec = tf.reshape(uv1, (shape[0], shape[1] * shape[2], 3))

    return uv1_vec


def epipolar_errors(predict_fundamental_matrix_in, flow, bin_size=-1):
    """
    return: a tensor with shape (num_batchs, height*width) with the epipolar errors of the flow given the
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

    flow_bs, flow_h, flow_w, two = flow.shape.as_list()
    assert (two == 2)
    assert (flow_bs == batch_size)

    # Get image point coordinates as flow vector in homogenous coordinates.
    old_points = get_image_coordinates_as_points((batch_size, flow_h, flow_w))
    old_points = tf.transpose(old_points, (0, 2, 1))

    # Add calculated flow to images coordinates to get new flow.
    flow_vec = tf.reshape(flow, (batch_size, flow_h * flow_w, two))
    flow_vec = tf.transpose(flow_vec, (0, 2, 1))
    flow_vec = tf.concat((flow_vec, tf.zeros((batch_size, 1, flow_h * flow_w))), axis=1)
    new_points = old_points + flow_vec
    #
    # if bin_size > 0:
    #     us = [i * bin_size for i in range(int(flow_h * flow_w / bin_size))]
    #     old_points = tf.gather(old_points, us, axis=1)
    #     new_points = tf.gather(new_points, us, axis=1)

    add_to_summary('debug/old_points', old_points)
    add_to_summary('debug/new_points', new_points)
    add_to_summary('debug/pred', pred_fun)

    def print_nan_tf(t, msg=""):
        cond = tf.reduce_any(tf.is_nan(t))
        return tf.Print(t, [msg, cond])

    # def assert_nan_tf(t):
    #     cond = tf.reduce_any(tf.is_nan(t))
    #     return tf.assert_equal(cond, tf.constant([True]))

    old_points = print_nan_tf(old_points, "old")
    new_points = print_nan_tf(new_points, "new")
    pred_fun = tf.Print(pred_fun,
                        [tf.reduce_max(pred_fun), tf.reduce_min(pred_fun), tf.reduce_mean(pred_fun), pred_fun])

    # Calculate epipolar error.
    tmp = tf.matmul(pred_fun, old_points)

    # Get errors.
    # This is like tf.matrix_diag_part(tf.matmul(new_points, tmp, transpose_a=True)) but with less memory usage.
    # error_vec = tf.einsum('aij,aij->aj', new_points, tmp)  # should be the same
    error_vec = tf.reduce_sum(tf.multiply(new_points, tmp), axis=2)
    error_vec = print_nan_tf(error_vec, 'err_v')
    add_to_summary('debug/error', error_vec)
    return error_vec


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


def add_to_summary(name, tensor):
    tf.summary.scalar(name + '/mean', tf.reduce_mean(tensor))
    tf.summary.scalar(name + '/max', tf.reduce_max(tensor))
    tf.summary.scalar(name + '/min', tf.reduce_min(tensor))
