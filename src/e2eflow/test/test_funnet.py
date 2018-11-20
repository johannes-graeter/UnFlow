import unittest
# from .funnet import FunNet
from ..core.util import epipolar_errors
from ..core.util import get_fundamental_matrix as get_fun_tf
import numpy as np
# from matplotlib import pyplot as plt
import tensorflow as tf


def get_rotation(angle, axis=0):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 1:
        return np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 2:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        raise Exception("Wrong axis defined.")


def get_test_flow(width, height, rotation, translation, intrin):
    depth = 20. * np.ones((height, width)) + 0. * (np.random.rand(height, width) - 0.5)

    intrin_inv = np.linalg.inv(intrin)

    flow = np.zeros((height, width, 2))
    for u in range(width):
        for v in range(height):
            img_point_old = np.array([u, v, 1.])

            p = depth[v, u] * intrin_inv.dot(img_point_old)
            p_new = rotation.dot(p) + translation
            img_point_new = intrin.dot(p_new / p_new[2])
            flow[v, u, :] = (img_point_new - img_point_old)[:2]

    return flow


def get_fundamental_matrix(rotation, t, intrin):
    cross_t = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = cross_t.dot(rotation)  # as in multiple view geometry p.257
    intrin_inv = np.linalg.inv(intrin)
    F = (intrin_inv.transpose().dot(E)).dot(intrin_inv)
    return F


"""
def plot(flow):
    fig, ax = plt.subplots(2)
    cf0 = ax[0].imshow(flow[:, :, 0])
    cf1 = ax[1].imshow(flow[:, :, 1])
    fig.colorbar(cf0, ax=ax[0])
    fig.colorbar(cf1, ax=ax[1])
    ax[0].set_title("Flow u")
    ax[1].set_title("Flow v")
    plt.show()
"""


class TestEpipolarError(unittest.TestCase):
    f = 700.
    cu = 350.
    cv = 150.
    intrin = np.array([[f, 0., cu], [0., f, cv], [0., 0., 1.]])

    def test_synthetic(self):
        rotation = get_rotation(1. / 180. * np.pi, axis=1)
        translation = np.array([0., 0., 2.])
        flow = get_test_flow(5, 5, rotation, translation, self.intrin)
        flow_tf = tf.expand_dims(tf.convert_to_tensor(flow, np.float32), axis=0)

        F = get_fundamental_matrix(rotation, translation, self.intrin)
        # f must have shape (num_batches, 9, 1)
        f0 = tf.expand_dims(tf.convert_to_tensor(F.reshape(1, 9), np.float32), axis=2)

        errs0 = epipolar_errors(f0, flow_tf)

        f1 = tf.expand_dims(tf.convert_to_tensor(F, np.float32), axis=0)
        errs1 = epipolar_errors(f1, flow_tf)

        with tf.Session() as sess:
            a = tf.reduce_sum(errs0).eval()
            b = tf.reduce_sum(tf.squared_difference(errs0, errs1)).eval()

        self.assertLess(a, 1e-8)
        self.assertLess(b, 1e-8)

    def test_fundamental_matrix(self):
        r_p_y_ty_tp = np.array([0, 0, 5. / 180. * np.pi, 10. / 180. * np.pi, 0.])

        yaw = get_rotation(r_p_y_ty_tp[2], axis=1)
        roll = get_rotation(r_p_y_ty_tp[0], axis=2)
        pitch = get_rotation(r_p_y_ty_tp[1], axis=0)

        rotation = yaw.dot(pitch.dot(roll))

        t = np.array([0, 0, 1])
        t = get_rotation(r_p_y_ty_tp[3], axis=1).dot(t)
        t = get_rotation(r_p_y_ty_tp[4], axis=0).dot(t)
        F_np = get_fundamental_matrix(rotation, t, self.intrin)
        f1 = tf.convert_to_tensor(F_np, np.float32)

        angles_tf = tf.convert_to_tensor(r_p_y_ty_tp, np.float32)

        f2 = get_fun_tf(tf.stack([angles_tf, 0. * angles_tf]), tf.convert_to_tensor(self.intrin, np.float32))

        with tf.Session() as sess:
            a = tf.reduce_sum(tf.squared_difference(f1, f2[0, :, :])).eval()

        self.assertLess(a, 1e-7)


if __name__ == '__main__':
    unittest.main()
