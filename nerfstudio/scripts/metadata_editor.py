import argparse
import glob
import json
import os
from pathlib import Path
import yaml
import pandas
import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image


#Code to add sensor depth path to each frame
def add_sensor_depth(metadata_path):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        data["has_sensor_depth"] = True

        for frame in data["frames"]:
            frame["sensor_depth_path"] = frame["mono_depth_path"]

        f.seek(0)     
        json.dump(data, f, indent=4)
        f.truncate()


#Code to recenter frame positions
def recenter_frames(metadata_path):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        pose_list = []


        for frame in data["frames"]:
            pose_list.append(frame["camtoworld"])
        
        poses = np.stack(pose_list, axis=0)

        print(poses.shape)
        

        min_vertices = poses[:, :3, 3].min(axis=0)
        max_vertices = poses[:, :3, 3].max(axis=0)

        #Increase scene_scale as we are working with indoor scene
        scene_scale_mult = 20.0
        # scene_scale_mult = 2.0

        #Calculate scene scale and center from poses
        scene_scale = 2.0 / (np.max(max_vertices - min_vertices) * scene_scale_mult)

        scene_center = (min_vertices + max_vertices) / 2.0

        # normalize pose to unit cube
        poses[:, :3, 3] -= scene_center
        poses[:, :3, 3] *= scene_scale

        for index, frame in enumerate(data["frames"]):
            frame["camtoworld"] = poses[index,:,:].tolist()

        # print(data["frames"])
        f.seek(0)     
        json.dump(data, f, indent=4)
        f.truncate()


#Code to add new paths to each frame
def add_new_paths(metadata_path):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        # data["has_sensor_depth"] = True

        for frame in data["frames"]:
            prev_rgb = frame["rgb_path"]
            prev_normal = frame["mono_normal_path"]
            prev_depth = frame["mono_depth_path"]

            frame["rgb_path"] = "rgb/" + prev_rgb
            frame["mono_normal_path"] = "normal/" + prev_normal
            frame["mono_depth_path"] = "depth/" + prev_depth


        f.seek(0)     
        json.dump(data, f, indent=4)
        f.truncate()

def change_intrinsics(metadata_path):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        # breakpoint()
        for frame in data["frames"]:
            old_intrinsics = frame["intrinsics"]
            frame["intrinsics"][0][2] = frame["intrinsics"][0][2] / (1271.0 / 384.0)
       
            print(frame["intrinsics"])

        # print(data["frames"])
        f.seek(0)     
        json.dump(data, f, indent=4)
        f.truncate()


def remove_stereo(metadata_path):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        new_frames = []

        # breakpoint()

        length = len(data["frames"])
        for idx, frame in enumerate(data["frames"]):
            if idx < length/2:
                # print(idx)
                # print(frame)
                new_frames.append(frame)


        # print(len(new_frames))
        data["frames"] = new_frames

        f.seek(0)     
        json.dump(data, f, indent=4)
        f.truncate()


#Convert sdf studio data convention to nerfstudio
def sdf_to_nerfstudio(metadata_path, output_path, change_camera_format):
    with open(metadata_path, 'r+') as f:
        data = json.load(f)

        new_data = {}

        first_frame_intrinsics = data["frames"][0]["intrinsics"]
        # print(first_frame_intrinsics)

        new_data["camera_model"] = "OPENCV"
        new_data["fl_x"] = first_frame_intrinsics[0][0]
        new_data["fl_y"] = first_frame_intrinsics[1][1]
        # new_data["k1"] = 0.0
        # new_data["k2"] = 0.0
        # new_data["p1"] = 0.0
        # new_data["p2"] = 0.0
        new_data["cx"] = first_frame_intrinsics[0][2]
        new_data["cy"] = first_frame_intrinsics[1][2]
        new_data["w"] = data["width"]
        new_data["h"] = data["height"]
        # new_data["aabb_scale"] = 1

        frame_list = []

        for idx, frame in enumerate(data["frames"]):

            new_frame = {}
            new_frame["file_path"] = frame["rgb_path"]

            #Uncomment the below line to add links to depth images
            # new_frame["depth_file_path"] = frame["mono_depth_path"].replace('depth.npy', 'depth_scaled.png')

            c2w = np.array(frame["camtoworld"]).reshape(4, 4)
            if change_camera_format:
                c2w[0:3, 1:3] *= -1
            new_frame["transform_matrix"] = c2w.tolist()

            frame_list.append(new_frame)   
        
        
        new_data["frames"] = frame_list

        # print(new_data)

        f.seek(0)
        with open(output_path + "/transforms.json", "w", encoding="utf-8") as o:
            json.dump(new_data, o, indent=4)     


metadata_path = '/home/casimir/ETH/nerfstudio/data/2011_09_26_drive_0001_sync_0/meta_data.json'
output_path = '/home/casimir/ETH/nerfstudio/data/2011_09_26_drive_0001_sync_0'


# add_sensor_depth(metadata_path)
# recenter_frames(metadata_path)
# add_new_paths(metadata_path)
# change_intrinsics(metadata_path)
# remove_stereo(metadata_path)
sdf_to_nerfstudio(metadata_path, output_path, change_camera_format=True)
