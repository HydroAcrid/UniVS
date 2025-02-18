import os
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_hurricane_metadata():
    meta = {
        "thing_classes": ["hurricane_damage"],  # adjust classes as needed
        "thing_colors": [(255, 0, 0)],  # one color per class
    }
    return meta

def register_hurricane_dataset(root):
    """
    Register HurricaneVidNet dataset
    Args:
        root: Path to dataset directory
    """
    root = os.path.join(root, "datasets/HurricaneVidNet_Dataset")
    
    # Register video dataset
    DatasetCatalog.register(
        "hurricane_vidnet_video",
        lambda: load_hurricane_video_data(root)
    )
    MetadataCatalog.get("hurricane_vidnet_video").set(
        **get_hurricane_metadata()
    )

def load_hurricane_video_data(root):
    """
    Load video data annotations
    """
    dataset_dicts = []
    # Add your video data loading logic here
    # Format should match Detectron2's expected format
    return dataset_dicts

from .hurricane_vidnet import register_hurricane_dataset

def register_all_hurricane_vidnet(root):
    register_hurricane_dataset(root)

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_hurricane_vidnet(_root)
