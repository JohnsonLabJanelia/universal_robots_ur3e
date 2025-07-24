import numpy as np
import csv
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", type=str, required=True)
parser.add_argument("-s", "--save_folder", type=str, required=True)

args = parser.parse_args()
label_file = args.label


def rig2mujoco_transform(coord):
    rig2mujoco_r = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    rig2mujoco_r = rig2mujoco_r.T  # c to python
    rig2mujoco_tv = np.asarray([0, 0, 0])  #
    new_coord = np.matmul(rig2mujoco_r, coord) + rig2mujoco_tv
    return list(new_coord / 1000)


def load_jarvis_3d_ball_csv(file_name, num_keypoints):
    labels = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if "NaN" not in row:
                    keypoints = [float(x) for x in row[2:]]
                    keypoints = np.asarray(keypoints)
                    keypoints = keypoints.reshape([num_keypoints, 3])
                    labels.append(keypoints)
            line_count += 1
    labels = np.concatenate(labels)
    return labels


labels = load_jarvis_3d_ball_csv(label_file, 1)

ball_calibration = 1000 * np.asarray(
    [
        [0.271902, 0.359436, 0.394134],
        [0.431698, 0.315648, 0.408148],
        [0.461963, 0.263889, 0.269816],
        [0.418017, 0.499603, 0.299859],
        [0.411305, 0.472677, 0.409867],
    ]
)


labels = labels.reshape((1, 5, 3))

# todo: fit rotation matrix
r_matrix_all = np.zeros([1, 3, 3])
r_matrix_all[0] = np.eye(3, 3)

t_vector_all = np.zeros([1, 3])
for robot_idx in range(1):
    labels_per_robot = labels[robot_idx]
    t_vec = ball_calibration.T - np.matmul(
        r_matrix_all[robot_idx], labels_per_robot.T
    )
    t_vec_mean = t_vec.mean(axis=1)
    t_vec_std_err = np.sqrt(t_vec.var(axis=1))
    print("t_vec mean: {}, std: {}".format(t_vec_mean, t_vec_std_err))
    t_vector_all[robot_idx, :] = t_vec_mean

## load red points, todo train a jarvirs model for 3d keypoints tracking
ball_diff = np.zeros([ball_calibration.shape[0] - 1])
for i in range(ball_calibration.shape[0] - 1):
    ball_diff[i] = np.linalg.norm(
        ball_calibration[i + 1] - ball_calibration[i]
    )

calib_results = []
for robot_idx in range(1):
    labels_per_robot = labels[robot_idx]
    to_compare = np.matmul(r_matrix_all[robot_idx], labels_per_robot.T).T
    labels_diff = np.zeros(ball_calibration.shape[0] - 1)
    for i in range(ball_calibration.shape[0] - 1):
        labels_diff[i] = np.linalg.norm(to_compare[i + 1] - to_compare[i])

    calib_results_per_robot = np.abs(ball_diff - labels_diff)
    print("Robot error {}: {}".format(robot_idx, calib_results_per_robot))
    calib_results.append(calib_results_per_robot)
    print(
        "Robox error {}: mean {:.3f}mm, std {:.3f}mm".format(
            robot_idx,
            calib_results_per_robot.mean(axis=0),
            np.sqrt(calib_results_per_robot.var()),
        )
    )


# save config
save_folder = args.save_folder
robot_names = ["URB3"]
for robot_idx in range(1):
    data = {
        "rotation_matrix": r_matrix_all[robot_idx].T.tolist(),
        "tvec": t_vector_all[robot_idx].tolist(),
    }
    to_save = "{}/{}.json".format(save_folder, robot_names[robot_idx])
    with open(to_save, "w") as f:
        json.dump(data, f)


with open(to_save) as f:
    robot_config = json.load(f)
# get rig space tvec
robot_rotation_matrix = np.asarray(robot_config["rotation_matrix"]).T
rig_robot = np.matmul(robot_rotation_matrix.T, robot_config["tvec"]) * (-1.0)

pos = rig2mujoco_transform(rig_robot)
angle = np.array([0, 0, 1, np.pi])
s = f'<body pos="{pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}" axisangle="{angle[0]:.8f} {angle[1]:.8f} {angle[2]:.8f} {angle[3]:.8f}"/>'
print(s)
