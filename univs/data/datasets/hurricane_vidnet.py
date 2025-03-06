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

def register_hurricane_dataset(root):
    """
    Register HurricaneVidNet dataset
    Args:
        root: Path to dataset directory
    """
    # Check if already registered to avoid duplicate registration
    if "hurricane_vidnet_video" in DatasetCatalog:
        return
    
    DatasetCatalog.register(
        "hurricane_vidnet_video",
        lambda: load_hurricane_video_data(root)
    )
    MetadataCatalog.get("hurricane_vidnet_video").set(
        **get_hurricane_metadata(),
        json_file=os.path.join("/data/datasets", "HurricaneVidNet_Dataset", "output.json"),
        evaluator_type="sem_seg",
    )

def load_hurricane_video_data(root):
    """
    Load video data annotations
    Args:
        root: Path to dataset directory
    Returns:
        list[dict]: List of dictionaries in Detectron2 format
    """
    json_file = os.path.join("/data/datasets", "HurricaneVidNet_Dataset", "output.json")
    images_dir = os.path.join("/data/datasets", "HurricaneVidNet_Dataset", "images")
    
    with PathManager.open(json_file) as f:
        json_info = json.load(f)

    # Create category id mapper
    cat_ids = [cat["id"] for cat in json_info["categories"]]
    cat_id_map = {id: idx for idx, id in enumerate(cat_ids)}
    
    dataset_dicts = []
    for image_dict in json_info["images"]:
        record = {}
        
        # Update file_name to match your directory structure
        file_name = os.path.join(images_dir, image_dict["file_name"])
        record["file_name"] = file_name
        record["height"] = image_dict["height"]
        record["width"] = image_dict["width"]
        record["image_id"] = image_dict["id"]
        
        # Find annotations for this image
        annos = [
            anno for anno in json_info["annotations"] 
            if anno["image_id"] == image_dict["id"]
        ]
        
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "segmentation": anno["segmentation"],
                "category_id": cat_id_map[anno["category_id"]],
            }
            objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

from .hurricane_vidnet import register_hurricane_dataset

def register_all_hurricane_vidnet(root):
    register_hurricane_dataset(root)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hurricane_vidnet(_root)
