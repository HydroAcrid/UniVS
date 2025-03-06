import os
from detectron2.data import DatasetCatalog, MetadataCatalog
import json
from pathlib import Path
from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager

def get_hurricane_metadata():
    meta = {
        "thing_classes": [
            "Building-Total-Destruction",
            "Building-Major-Damage",
            "Building-Minor-Damage",
            "Building-No-Damage"
        ],
        "thing_colors": [
            (220, 20, 60),   # red for total destruction
            (255, 140, 0),   # orange for major damage
            (255, 255, 0),   # yellow for minor damage
            (0, 255, 0),     # green for no damage
        ]
    }
    return meta

def register_hurricane_dataset(root, debug=True):
    """
    Register HurricaneVidNet dataset
    Args:
        root: Path to dataset directory
    """
    # Check if already registered to avoid duplicate registration
    if "hurricane_vidnet_video" in DatasetCatalog:
        return
    
    output_dir = os.path.join(os.path.expanduser("~"), "hurricane_data")
    os.makedirs(output_dir, exist_ok=True)
    ytvis_json_file = os.path.join(output_dir, "ytvis_format.json")
    
    DatasetCatalog.register(
        "hurricane_vidnet_video",
        lambda: load_hurricane_video_data(root, output_dir, debug=debug)
    )
    
    MetadataCatalog.get("hurricane_vidnet_video").set(
        **get_hurricane_metadata(),
        json_file=ytvis_json_file,
        evaluator_type="ytvis",
    )

def load_hurricane_video_data(root, output_dir=None, debug=True):
    """
    Load video data annotations
    Args:
        root: Path to dataset directory
        output_dir: Directory to save converted data
        debug: Whether to run in debug mode with extra validation
    Returns:
        list[dict]: List of dictionaries in Detectron2 format
    """
    json_file = os.path.join("/data/datasets", "HurricaneVidNet_Dataset", "output.json")
    images_dir = os.path.join("/data/datasets", "HurricaneVidNet_Dataset", "images")
    
    # Use a directory where you have write permissions
    if output_dir is None:
        output_dir = os.path.join(os.path.expanduser("~"), "hurricane_data")
    os.makedirs(output_dir, exist_ok=True)
    
    ytvis_json_file = os.path.join(output_dir, "ytvis_format.json")
    
    with PathManager.open(json_file) as f:
        coco_data = json.load(f)
    
    # Convert COCO format to YTVIS format
    ytvis_data = convert_coco_to_ytvis(coco_data)
    
    # Validate the converted data
    if not validate_ytvis_format(ytvis_data):
        print("WARNING: YTVIS format data validation failed!")
        # You could raise an exception here if you want to stop execution
        # raise ValueError("YTVIS format data validation failed")
    
    # Save the converted data for debugging/reference
    with open(ytvis_json_file, 'w') as f:
        json.dump(ytvis_data, f)
    
    # Create category id mapper
    cat_ids = [cat["id"] for cat in ytvis_data["categories"]]
    cat_id_map = {id: idx for idx, id in enumerate(cat_ids)}
    
    dataset_dicts = []
    
    # Process each video
    for video in ytvis_data["videos"]:
        video_id = video["id"]
        height = video["height"]
        width = video["width"]
        
        # Create a record for the video
        record = {
            "file_names": [os.path.join(images_dir, fname) for fname in video["file_names"]],
            "height": height,
            "width": width,
            "video_id": video_id,
            "length": len(video["file_names"]),
            "dataset_name": "hurricane_vidnet_video"
        }
        
        # Find annotations for this video
        video_annos = [anno for anno in ytvis_data["annotations"] if anno["video_id"] == video_id]
        
        instances = []
        for anno in video_annos:
            instance = {
                "category_id": cat_id_map[anno["category_id"]],
                "segmentations": anno["segmentations"],
                "bboxes": anno["bboxes"],
                "areas": anno["areas"]
            }
            instances.append(instance)
        
        record["annotations"] = instances
        dataset_dicts.append(record)
    
    if debug:
        # Print more detailed information
        print(f"Found {len(ytvis_data['videos'])} videos")
        print(f"Found {len(ytvis_data['annotations'])} annotations")
        print(f"Found {len(ytvis_data['categories'])} categories")
        
        # Validate the converted data
        validate_ytvis_format(ytvis_data)
        
        # After creating dataset_dicts
        inspect_sample_data(dataset_dicts)
    
    return dataset_dicts

from .hurricane_vidnet import register_hurricane_dataset

def register_all_hurricane_vidnet(root):
    register_hurricane_dataset(root)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hurricane_vidnet(_root)

def convert_coco_to_ytvis(coco_data):
    """
    Convert COCO format to YTVIS format
    Args:
        coco_data: Dictionary containing COCO format data
    Returns:
        Dictionary in YTVIS format
    """
    ytvis_data = {
        "categories": coco_data["categories"],
        "videos": [],
        "annotations": []
    }
    
    # Group images by video
    video_frames = {}
    for image in coco_data["images"]:
        # Extract video ID from filename (e.g., "Hurricane_Ian_10012022_Palmeto_Palms-2/0000007.png")
        video_id = image["file_name"].split("/")[0]
        if video_id not in video_frames:
            video_frames[video_id] = []
        video_frames[video_id].append(image)
    
    # Create videos
    video_id_map = {}  # Map from video name to numeric ID
    for idx, (video_name, frames) in enumerate(video_frames.items(), 1):
        sorted_frames = sorted(frames, key=lambda x: x["id"])
        ytvis_data["videos"].append({
            "id": idx,
            "name": video_name,
            "width": sorted_frames[0]["width"],
            "height": sorted_frames[0]["height"],
            "file_names": [frame["file_name"] for frame in sorted_frames]
        })
        video_id_map[video_name] = idx
    
    # Group annotations by instance and video
    instance_anns = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        # Find the image to get the video name
        for image in coco_data["images"]:
            if image["id"] == image_id:
                video_name = image["file_name"].split("/")[0]
                instance_key = f"{video_name}_{ann['category_id']}_{ann['id']}"
                if instance_key not in instance_anns:
                    instance_anns[instance_key] = {
                        "video_name": video_name,
                        "category_id": ann["category_id"],
                        "segmentations": {},
                        "areas": {},
                        "bboxes": {}
                    }
                # Find frame index within the video
                frame_idx = image["file_name"].split("/")[1].split(".")[0]
                
                # Ensure segmentation is a list of lists (polygon format)
                if isinstance(ann["segmentation"], str):
                    # If it's a string, convert to proper format or set to None
                    instance_anns[instance_key]["segmentations"][frame_idx] = None
                else:
                    instance_anns[instance_key]["segmentations"][frame_idx] = ann["segmentation"]
                
                # Ensure area is a number
                if isinstance(ann["area"], (int, float)):
                    instance_anns[instance_key]["areas"][frame_idx] = ann["area"]
                else:
                    instance_anns[instance_key]["areas"][frame_idx] = 0
                
                # Ensure bbox is a list of numbers
                if isinstance(ann["bbox"], list) and len(ann["bbox"]) == 4:
                    instance_anns[instance_key]["bboxes"][frame_idx] = ann["bbox"]
                else:
                    instance_anns[instance_key]["bboxes"][frame_idx] = [0, 0, 0, 0]
                
                break
    
    # Create video annotations
    for idx, (_, ann_data) in enumerate(instance_anns.items(), 1):
        video_id = video_id_map[ann_data["video_name"]]
        
        # Get all frames for this video
        video_frames = next(v["file_names"] for v in ytvis_data["videos"] if v["id"] == video_id)
        num_frames = len(video_frames)
        
        # Create lists for all frames (None for frames without annotations)
        segmentations = [None] * num_frames
        areas = [0] * num_frames  # Use 0 instead of None for areas
        bboxes = [[0, 0, 0, 0]] * num_frames  # Use empty bbox instead of None
        
        # Fill in the frames that have annotations
        for frame_idx, segm in ann_data["segmentations"].items():
            # Convert frame_idx (string like "0000001") to integer index
            try:
                idx_num = int(frame_idx)
                if idx_num < num_frames:
                    segmentations[idx_num] = segm
                    areas[idx_num] = ann_data["areas"].get(frame_idx, 0)
                    bboxes[idx_num] = ann_data["bboxes"].get(frame_idx, [0, 0, 0, 0])
            except ValueError:
                # Skip if frame_idx can't be converted to int
                continue
        
        ytvis_data["annotations"].append({
            "id": idx,
            "video_id": video_id,
            "category_id": ann_data["category_id"],
            "segmentations": segmentations,
            "areas": areas,
            "bboxes": bboxes
        })
    
    return ytvis_data

def validate_ytvis_format(ytvis_data):
    """
    Validate YTVIS format data to catch common errors
    Args:
        ytvis_data: Dictionary in YTVIS format
    Returns:
        bool: True if valid, False otherwise
    """
    print("Validating YTVIS format data...")
    is_valid = True
    
    # Check required keys
    required_keys = ["categories", "videos", "annotations"]
    for key in required_keys:
        if key not in ytvis_data:
            print(f"❌ Missing required key: {key}")
            is_valid = False
    
    # Check videos
    if "videos" in ytvis_data:
        for i, video in enumerate(ytvis_data["videos"]):
            required_video_keys = ["id", "name", "width", "height", "file_names"]
            for key in required_video_keys:
                if key not in video:
                    print(f"❌ Video {i} missing required key: {key}")
                    is_valid = False
            
            # Check file_names is a list
            if "file_names" in video and not isinstance(video["file_names"], list):
                print(f"❌ Video {i} file_names is not a list")
                is_valid = False
            elif "file_names" in video and len(video["file_names"]) == 0:
                print(f"❌ Video {i} has no frames")
                is_valid = False
    
    # Check annotations
    if "annotations" in ytvis_data:
        for i, ann in enumerate(ytvis_data["annotations"]):
            required_ann_keys = ["id", "video_id", "category_id", "segmentations", "areas", "bboxes"]
            for key in required_ann_keys:
                if key not in ann:
                    print(f"❌ Annotation {i} missing required key: {key}")
                    is_valid = False
            
            # Check segmentations, areas, bboxes are lists of the same length
            if all(k in ann for k in ["segmentations", "areas", "bboxes"]):
                if not isinstance(ann["segmentations"], list):
                    print(f"❌ Annotation {i} segmentations is not a list")
                    is_valid = False
                if not isinstance(ann["areas"], list):
                    print(f"❌ Annotation {i} areas is not a list")
                    is_valid = False
                if not isinstance(ann["bboxes"], list):
                    print(f"❌ Annotation {i} bboxes is not a list")
                    is_valid = False
                
                # Check all lists have the same length
                lengths = [len(ann["segmentations"]), len(ann["areas"]), len(ann["bboxes"])]
                if len(set(lengths)) > 1:
                    print(f"❌ Annotation {i} has inconsistent lengths: {lengths}")
                    is_valid = False
                
                # Check segmentations format
                for j, segm in enumerate(ann["segmentations"]):
                    if segm is not None and not isinstance(segm, list):
                        print(f"❌ Annotation {i}, segmentation {j} is not None or a list")
                        is_valid = False
                
                # Check bboxes format
                for j, bbox in enumerate(ann["bboxes"]):
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        print(f"❌ Annotation {i}, bbox {j} is not a list of 4 numbers")
                        is_valid = False
    
    if is_valid:
        print("✅ YTVIS format data is valid")
    return is_valid

def inspect_sample_data(dataset_dicts, num_samples=3):
    """
    Inspect a sample of the dataset to check for common issues
    Args:
        dataset_dicts: List of dataset dictionaries
        num_samples: Number of samples to inspect
    """
    print(f"\nInspecting {num_samples} random samples from dataset...")
    import random
    
    samples = random.sample(dataset_dicts, min(num_samples, len(dataset_dicts)))
    
    for i, d in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"  Video ID: {d.get('video_id', 'MISSING')}")
        print(f"  Length: {d.get('length', 'MISSING')}")
        print(f"  Height x Width: {d.get('height', 'MISSING')} x {d.get('width', 'MISSING')}")
        print(f"  Dataset name: {d.get('dataset_name', 'MISSING')}")
        
        # Check file_names
        if "file_names" in d:
            print(f"  Number of frames: {len(d['file_names'])}")
            print(f"  First frame: {d['file_names'][0]}")
            # Check if files exist
            if not os.path.exists(d['file_names'][0]):
                print(f"  ❌ First frame file does not exist!")
        else:
            print("  ❌ Missing file_names!")
        
        # Check annotations
        if "annotations" in d:
            print(f"  Number of instances: {len(d['annotations'])}")
            if len(d['annotations']) > 0:
                ann = d['annotations'][0]
                print(f"  First instance category: {ann.get('category_id', 'MISSING')}")
                print(f"  Has segmentations: {('segmentations' in ann)}")
                print(f"  Has bboxes: {('bboxes' in ann)}")
                
                # Check segmentations and bboxes
                if 'segmentations' in ann:
                    num_valid_segms = sum(1 for s in ann['segmentations'] if s is not None)
                    print(f"  Valid segmentations: {num_valid_segms}/{len(ann['segmentations'])}")
                
                if 'bboxes' in ann:
                    num_valid_bboxes = sum(1 for b in ann['bboxes'] if b != [0, 0, 0, 0])
                    print(f"  Valid bboxes: {num_valid_bboxes}/{len(ann['bboxes'])}")
        else:
            print("  ❌ Missing annotations!")
    
    print("\nSample inspection complete.")
