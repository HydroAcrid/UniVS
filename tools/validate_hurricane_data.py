import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2

def validate_hurricane_dataset():
    # Get a sample from your dataset
    dataset_dicts = DatasetCatalog.get("hurricane_vidnet_video")
    metadata = MetadataCatalog.get("hurricane_vidnet_video")
    
    # Check required fields
    for idx, d in enumerate(dataset_dicts):
        print(f"\nValidating sample {idx}:")
        
        # Check required fields
        required_fields = ["file_name", "height", "width", "image_id"]
        for field in required_fields:
            if field not in d:
                print(f"❌ Missing required field: {field}")
            else:
                print(f"✅ Has required field: {field}")
        
        # Verify image exists
        if os.path.exists(d["file_name"]):
            print(f"✅ Image file exists: {d['file_name']}")
            # Load image to verify it's readable
            img = cv2.imread(d["file_name"])
            if img is not None:
                print(f"✅ Image is readable, shape: {img.shape}")
            else:
                print("❌ Image cannot be read!")
        else:
            print(f"❌ Image file does not exist: {d['file_name']}")
        
        # Check annotations format
        if "annotations" in d:
            for ann in d["annotations"]:
                required_ann_fields = ["bbox", "category_id", "segmentation"]
                for field in required_ann_fields:
                    if field not in ann:
                        print(f"❌ Annotation missing required field: {field}")
                    else:
                        print(f"✅ Annotation has required field: {field}")
        
        # Visualize first few samples
        if idx < 3:  # Show first 3 samples
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
            vis = visualizer.draw_dataset_dict(d)
            cv2.imwrite(f"validation_vis_{idx}.jpg", vis.get_image()[:, :, ::-1])
            print(f"✅ Saved visualization to validation_vis_{idx}.jpg")

if __name__ == "__main__":
    validate_hurricane_dataset()
