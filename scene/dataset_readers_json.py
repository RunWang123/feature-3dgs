# Modified dataset_readers.py to support JSON-based train/test split
# Add this function to dataset_readers.py or import from here

import json
import os

def load_split_from_json(json_path, scene_name):
    """
    Load train/test split from JSON file.
    
    Args:
        json_path: Path to JSON split file
        scene_name: Scene name (e.g., "scene0686_01")
    
    Returns:
        train_names: List of training image names (without extension)
        test_names: List of test image names (without extension)
    """
    with open(json_path, 'r') as f:
        split_data = json.load(f)
    
    if scene_name not in split_data['scenes']:
        raise ValueError(f"Scene {scene_name} not found in JSON file")
    
    scene_data = split_data['scenes'][scene_name]
    
    # Collect all ref_views (training) and target_views (test)
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
    
    return train_names, test_names


def readColmapSceneInfo_withJSON(path, foundation_model, images, eval, 
                                  json_split_path=None, llffhold=8):
    """
    Modified readColmapSceneInfo that supports JSON-based splits.
    
    Args:
        path: Dataset path
        foundation_model: 'lseg' or 'sam'
        images: Images subfolder name
        eval: Whether to use eval split
        json_split_path: Path to JSON file with train/test split (optional)
        llffhold: Default split parameter (used if json_split_path is None)
    """
    from scene.dataset_readers import (
        read_extrinsics_binary, read_intrinsics_binary,
        read_extrinsics_text, read_intrinsics_text,
        readColmapCameras, getNerfppNorm, fetchPly, storePly,
        read_points3D_binary, read_points3D_text,
        BasicPointCloud, SceneInfo
    )
    
    # Load COLMAP data
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
    
    if foundation_model == 'sam':
        semantic_feature_dir = "sam_embeddings" 
    elif foundation_model == 'lseg':
        semantic_feature_dir = "rgb_feature_langseg"
    
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, 
        cam_intrinsics=cam_intrinsics, 
        images_folder=os.path.join(path, reading_dir), 
        semantic_feature_folder=os.path.join(path, semantic_feature_dir)
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)
    
    semantic_feature_dim = cam_infos[0].semantic_feature.shape[0]
    
    # Train/test split logic
    if eval:
        if json_split_path and os.path.exists(json_split_path):
            # Use JSON-based split
            scene_name = os.path.basename(path)
            train_names, test_names = load_split_from_json(json_split_path, scene_name)
            
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
            
        else:
            # Use default llffhold split
            train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 2]
            test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 2]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    # Load 3D points
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
    
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
        semantic_feature_dim=semantic_feature_dim
    )
    
    return scene_info

