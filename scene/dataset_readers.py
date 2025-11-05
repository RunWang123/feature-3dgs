#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    semantic_feature: torch.tensor 
    semantic_feature_path: str 
    semantic_feature_name: str 


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    semantic_feature_dim: int 

def load_split_from_json(json_path, scene_name, case_id=-1):
    """
    Load train/test split from JSON file.
    
    Args:
        json_path: Path to JSON split file
        scene_name: Scene name (e.g., "scene0686_01")
        case_id: Case ID to select (0-indexed). If -1, use all cases.
    
    Returns:
        train_names: List of training image names (without extension)
        test_names: List of test image names (without extension)
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    if scene_name not in split_data['scenes']:
        raise ValueError(f"Scene {scene_name} not found in JSON file")
    
    scene_data = split_data['scenes'][scene_name]
    
    # If case_id is specified, use only that case
    if case_id >= 0:
        if case_id >= len(scene_data):
            raise ValueError(f"Case ID {case_id} out of range. Scene has {len(scene_data)} cases (0-{len(scene_data)-1})")
        
        # Use only the specified case
        case = scene_data[case_id]
        train_names = case.get('ref_views', [])
        test_names = case.get('target_views', [])
        
        print(f"  Using case {case_id} only (out of {len(scene_data)} total cases)")
    else:
        # Collect all ref_views (training) and target_views (test) from all cases
        train_names = []
        test_names = []
        
        for case in scene_data:
            if 'ref_views' in case:
                train_names.extend(case['ref_views'])
            if 'target_views' in case:
                test_names.extend(case['target_views'])
        
        # Remove duplicates while preserving order
        train_names = list(dict.fromkeys(train_names))
        test_names = list(dict.fromkeys(test_names))
        
        print(f"  Using all {len(scene_data)} cases")
    
    return train_names, test_names

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, semantic_feature_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}, skipping camera {uid}")
            continue
            
        image = Image.open(image_path) 

        # Check if semantic feature exists
        semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
        if not os.path.exists(semantic_feature_path):
            print(f"Warning: LSeg feature not found: {semantic_feature_path}, skipping camera {uid}")
            continue
            
        semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
        semantic_feature = torch.load(semantic_feature_path) 

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name) 
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, foundation_model, images, eval, llffhold=8, json_split_path=None, case_id=-1):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images

    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir), semantic_feature_folder=os.path.join(path, semantic_feature_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    ###cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split('.')[0])) ### if img name is number
    # cam_infos =cam_infos[:30] ###: for scannet only
    # print(cam_infos)
    semantic_feature_dim = cam_infos[0].semantic_feature.shape[0]

    if eval:
        # Check if JSON split file is provided
        if json_split_path and os.path.exists(json_split_path):
            # Use JSON-based split
            scene_name = os.path.basename(path)
            train_names, test_names = load_split_from_json(json_split_path, scene_name, case_id)
            
            print(f"Using JSON split for {scene_name}:")
            print(f"  Train images: {len(train_names)}")
            print(f"  Test images:  {len(test_names)}")
            
            # Create lookup for faster matching
            train_names_set = set(train_names)
            test_names_set = set(test_names)
            
            train_cam_infos = []
            test_cam_infos = []
            
            for cam in cam_infos:
                # Remove file extension from image name for comparison
                img_name_no_ext = os.path.splitext(cam.image_name)[0]
                
                if img_name_no_ext in train_names_set:
                    train_cam_infos.append(cam)
                elif img_name_no_ext in test_names_set:
                    test_cam_infos.append(cam)
                # Images not in JSON are ignored
            
            print(f"  Loaded train: {len(train_cam_infos)}")
            print(f"  Loaded test:  {len(test_cam_infos)}")
            
            # Debug output if no cameras loaded
            if len(train_cam_infos) == 0 and len(test_cam_infos) == 0:
                print(f"  ⚠️  WARNING: No cameras matched JSON split!")
                available_names = [os.path.splitext(cam.image_name)[0] for cam in cam_infos]
                print(f"  Expected train frames: {sorted(list(train_names))[:10]}")
                print(f"  Expected test frames: {sorted(list(test_names))[:10]}")
                print(f"  Available frames: {sorted(list(available_names))[:10]}")
            
            # Validate that we loaded at least some cameras
            if len(train_cam_infos) == 0:
                raise ValueError(
                    f"No training cameras loaded for case {case_id}! "
                    f"This usually means the frame names in the JSON split don't match "
                    f"the actual image/feature files in the scene directory. "
                    f"Check {path}/images/ and {path}/rgb_feature_langseg/"
                )
        else:
            # Use default llffhold split
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2] # avoid 1st to be test view
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2] 
            # for i, item in enumerate(test_cam_infos): ### check test set
            #     print('test image:', item[7])
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Final validation: ensure we have training cameras
    if len(train_cam_infos) == 0:
        raise ValueError(
            f"No training cameras available! Cannot train 3DGS without any training views."
        )

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
            
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, semantic_feature_folder, extension=".png"): 
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            
            semantic_feature_path = os.path.join(semantic_feature_folder, image_name) + '_fmap_CxHxW.pt' 
            semantic_feature_name = os.path.basename(semantic_feature_path).split(".")[0]
            semantic_feature = torch.load(semantic_feature_path)
            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                              semantic_feature=semantic_feature,
                              semantic_feature_path=semantic_feature_path,
                              semantic_feature_name=semantic_feature_name)) 
            
    return cam_infos

def readNerfSyntheticInfo(path, foundation_model, white_background, eval, extension=".png"): 
    if foundation_model =='sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model =='lseg':
        semantic_feature_dir = "rgb_feature_langseg" 

    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, semantic_feature_folder=os.path.join(path, semantic_feature_dir)) 
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    semantic_feature_dim = train_cam_infos[0].semantic_feature.shape[0] 
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           semantic_feature_dim=semantic_feature_dim) 
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}