import numpy as np
import csv
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--label", type=str, required=True)
parser.add_argument("-s", "--save_folder", type=str, required=True)

args = parser.parse_args()
label_file = args.label


def load_jarvis_3d_ball_csv(file_name, num_keypoints):
    labels = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if "NaN" not in row:
                    keypoints = [float(x) for x in row[2:-1]]
                    keypoints = np.asarray(keypoints)
                    keypoints = keypoints.reshape([num_keypoints, 3])
                    labels.append(keypoints)
            line_count += 1
    labels = np.concatenate(labels)
    return labels


labels = load_jarvis_3d_ball_csv(label_file, 1)

ball_calibration = 1000 * np.asarray(
    [
        [-0.6, -0.6, 0.08],
        [0.0, -0.6, 0.08],
        [0.6, -0.6, 0.08],
        [0.6, -1.2, 0.08],
        [-0.0, -1.2, 0.08],
        [-0.6, -1.2, 0.08],
        [-0.6, -1.2, 0.18],
        [-0.6, -1.2, 0.28],
        [-0.6, -1.2, 0.38],
    ]
)


labels = labels.reshape((3, 9, 3))

# todo: fit rotation matrix
r_matrix_all = np.zeros([3, 3, 1])
r_matrix_all[0] = np.eye(3, 3)

t_vector_all = np.zeros([3, 1])
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
for robot_idx in range(3):
    data = {
        "rotation_matrix": r_matrix_all[robot_idx].T.tolist(),
        "tvec": t_vector_all[robot_idx].tolist(),
    }
    to_save = "{}/{}.json".format(save_folder, robot_names[robot_idx])
    with open(to_save, "w") as f:
        json.dump(data, f)
