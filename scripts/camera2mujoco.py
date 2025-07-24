import numpy as np
import pickle as pkl
import cv2
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--yaml_dir", type=str, required=True)
parser.add_argument(
    "-s", "--sensor_size", type=float, required=True, help="sensor size in mm"
)
args = parser.parse_args()

cam_names = []
for file in glob.glob(args.yaml_dir + "/*.yaml"):
    file_name = file.split("/")
    cam_names.append(file_name[-1][:-5])
cam_names.sort()

for which_camera in cam_names:
    cam_params = {}
    filename = args.yaml_dir + "/{}.yaml".format(which_camera)
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    cam_params["camera_matrix"] = fs.getNode("camera_matrix").mat()
    cam_params["distortion_coefficients"] = fs.getNode(
        "distortion_coefficients"
    ).mat()
    cam_params["tc_ext"] = fs.getNode("tc_ext").mat()
    cam_params["rc_ext"] = fs.getNode("rc_ext").mat()
    cam_params["image_width"] = int(fs.getNode("image_width").real())
    cam_params["image_height"] = int(fs.getNode("image_height").real())
    resolution = [cam_params["image_width"], cam_params["image_height"]]
    focalpixel = [
        cam_params["camera_matrix"][0, 0],
        cam_params["camera_matrix"][1, 1],
    ]
    sensorsize = [item * args.sensor_size for item in resolution]

    # extrinsics
    rotation = cam_params["rc_ext"].T
    translation = -np.matmul(rotation, cam_params["tc_ext"][:, 0])

    def rig2mujoco_transform(coord):
        rig2mujoco_r = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        rig2mujoco_r = rig2mujoco_r.T  # c to python
        rig2mujoco_tv = np.asarray([0, 0, 0])
        new_coord = np.matmul(rig2mujoco_r, coord) + rig2mujoco_tv
        return list(new_coord / 1000)

    t_mujoco = rig2mujoco_transform(translation)

    x_axis_world = np.matmul(rotation, np.array([1, 0, 0]))
    rig2mujoco_r = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rig2mujoco_r = rig2mujoco_r.T  # c to python
    x_aixs_rig = np.matmul(rig2mujoco_r, x_axis_world)
    x_aixs_rig = x_aixs_rig / np.linalg.norm(x_aixs_rig)
    # print(x_aixs_rig)

    y_axis_world = np.matmul(rotation, np.array([0, 1, 0]))
    y_aixs_rig = np.matmul(rig2mujoco_r, y_axis_world)
    y_aixs_rig = y_aixs_rig / np.linalg.norm(y_aixs_rig)
    # print(y_aixs_rig)
    # print(np.dot(x_aixs_rig, y_aixs_rig))

    xyaxes = np.concatenate([x_aixs_rig, -y_aixs_rig])
    mode = "fixed"
    camera_xml = (
        f'<camera name="{which_camera}" mode="{mode}" '
        f'focalpixel="{focalpixel[0]:.8f} {focalpixel[1]:.8f}" '
        f'resolution="{resolution[0]} {resolution[1]}" '
        f'sensorsize="{sensorsize[0]:.6f} {sensorsize[1]:.6f}" '
        f'pos="{t_mujoco[0]:.15f} {t_mujoco[1]:.15f} {t_mujoco[2]:.15f}" '
        f'xyaxes="{" ".join(f"{v:.8f}" for v in xyaxes)}"/>'
    )

    print(camera_xml)
