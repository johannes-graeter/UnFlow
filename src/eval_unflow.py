import glob
import os

import numpy as np


def get_rotation(angle, axis=0):
    if axis == 0:
        return np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
    elif axis == 1:
        return np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 2:
        return np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    else:
        raise Exception("Wrong axis defined.")


def make_transform(r_p_y_ty_tp):
    roll = get_rotation(r_p_y_ty_tp[0], axis=2)
    pitch = get_rotation(r_p_y_ty_tp[1], axis=0)
    yaw = get_rotation(r_p_y_ty_tp[2], axis=1)

    rotation = yaw.dot(pitch.dot(roll))

    t = np.array([0, 0, 1])
    t = get_rotation(r_p_y_ty_tp[3], axis=1).dot(t)
    t = get_rotation(r_p_y_ty_tp[4], axis=0).dot(t)

    return rotation, t


def to_affine(rot, trans):
    a = np.concatenate((rot, np.transpose(np.array([trans]))), axis=1)
    return np.concatenate((a, np.array([[0., 0., 0., 1.]])), axis=0)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="options for script")
    parser.add_argument("-p", "--path",
                        help="Path to folder with textfiles with angles as rotation and trasnlation direction.")
    parser.add_argument("-p5p", "--path_5_point",
                        help="Path to folder with textfiles with angles as rotation and trasnlation direction.")
    parser.add_argument("-l", "--lookup", help="Path to file with nunber to kitti sequence and calib")
    parser.add_argument("-o", "--output", help="Output folder.")
    parser.add_argument("-gt", "--groundtruth", help="groundtruth file path for scale and difference plotting.")

    parse_args = parser.parse_args()
    return parse_args


def accumulate_output(ids, output):
    # Accumulate output.
    acc = {}
    for id in ids:
        acc[id] = [np.eye(4)[:3, :4].flatten()]

    for id, motions in output.items():
        last = np.eye(4)
        for num, m in motions.items():
            last = last.dot(m)
            acc[id].append(last[:3, :4].flatten())
    return acc


def parse_calib_cam(inp):
    calib_mat = np.reshape(np.array(inp[1:]).astype(float), (3, 4))
    a = np.concatenate((np.eye(3), np.transpose(np.array([calib_mat[:, 3]])) * 1e-3), axis=1)
    a = np.concatenate((a, np.array([[0., 0., 0., 1.]])), axis=0)

    return a


def scale_result(output, ids, scales):
    output_scaled = {}
    for id in ids:
        output_scaled[id] = {}

    for key, scale_mat in scales.items():
        for row, motion in output[key].items():
            cur_motion = np.concatenate((motion[:3, :3], np.transpose(np.array([motion[:3, -1]])) * scale_mat[row]),
                                        axis=1)
            output_scaled[key][row] = np.concatenate((cur_motion, np.array([[0., 0., 0., 1.]])))

    return output_scaled


def main(args):
    # load calibs
    calib_paths = []
    with open(args.lookup, "r") as f:
        for l in f:
            calib_paths.append(l.strip("\n").split(" ")[-1])

    calibs = {}
    for p in set(calib_paths):
        with open(p.strip("\n"), "r") as f:
            ls = f.readlines()
            a = ls[-1].split(" ")
            b = ls[-2].split(" ")
            assert (a[0] == "P3:")
            assert (b[0] == "P2:")
            calibs[p] = {}
            calibs[p][a[0]] = parse_calib_cam(a)
            calibs[p][b[0]] = parse_calib_cam(b)

    ids = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11"]

    # Create output nested dict.
    output = {}
    for id in ids:
        output[id] = {}

    output_5p = {}
    for id in ids:
        output_5p[id] = {}

    with open(args.lookup, "r") as f:
        for line in f:
            num, sequ, calib_path = line.strip("\n").split(" ")

            # Get motion as affine.
            angles = np.loadtxt(os.path.join(args.path, "{}.txt".format(num)))
            rot, trans = make_transform(angles)
            motion = to_affine(rot, trans)

            motion_5point = np.loadtxt(os.path.join(args.path_5_point, "{}.txt".format(num)))
            motion_5_point = np.concatenate((np.reshape(motion_5_point, (3, 4)), np.array([[0., 0., 0., 1.]])), axis=0)

            # Transform to camera 0 (gray left)
            if "image_2" in sequ:
                id = 'P2:'
            elif "image_3" in sequ:
                id = 'P3:'
            else:
                raise Exception("Wrong calib")

            extr = calibs[calib_path][id]  # extrinsics from cam0 to cam1
            motion_cam_0 = np.linalg.inv(extr).dot(motion.dot(extr))
            motion_5point_cam0 = np.linalg.inv(extr).dot(motion_5point.dot(extr))

            sequ_num = sequ.split("/")
            for id in ids:
                if id in sequ_num:
                    num = int(sequ_num[-1].strip(".png"))
                    output[id][num] = motion_cam_0
                    output_5p[id][num] = motion_5point_cam0

    # Read groundtruth
    gt = {}
    for file in glob.glob(args.groundtruth + "/*.txt"):
        id = file.split("/")[-1].strip(".txt")
        assert (id in ids)
        p = np.reshape(np.loadtxt(file), (-1, 3, 4))
        a = np.reshape(np.tile(np.array([[0., 0., 0., 1.]]), p.shape[0]), (-1, 1, 4))
        gt[id] = np.concatenate((p, a), axis=1)

    gt_deltas = {}
    for id in ids:
        gt_deltas[id] = {}

    scales = {}
    for key, poses in gt.items():
        poses_inv = np.linalg.inv(poses)
        poses_inv = poses_inv[:-1, :, :]
        poses_inv = np.concatenate((np.array([np.eye(4)]), poses_inv), axis=0)
        deltas = np.matmul(poses_inv, poses)

        # Save delta poses.
        for i in range(deltas.shape[0]):
            gt_deltas[key][i] = deltas[i, :, :]

        # Get scales from deltas
        scales[key] = np.linalg.norm(deltas[:, :3, -1], axis=1)

    # Add scales to output.
    output_scaled = scale_result(output, ids, scales)
    acc = accumulate_output(ids, output_scaled)

    output_scaled_5point = scale_result(output_5p, ids, scales)
    acc_5p = accumulate_output(ids, output_scaled_5point)
    # acc_test = accumulate_output(ids, gt_deltas)

    # Dump it.
    for key, data in acc.items():
        np.savetxt(args.output + "/result_{}.txt".format(key), np.array(data))

    for key, data in acc_5p.items():
        np.savetxt(args.output + "/result_5p_{}.txt".format(key), np.array(data))

        # for key, data in acc_test.items():
        #     np.savetxt(args.output + "/test_gt_{}.txt".format(key), np.array(data))


if __name__ == '__main__':
    main(parse_args())
