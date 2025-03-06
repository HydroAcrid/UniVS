import os
import json
import argparse
import cv2

def parse_args():
    parser = argparse.ArgumentParser("hurricane dataset to coco format")
    parser.add_argument("--video_dir", default="datasets/custom_videos/raw/hurricane_videos", type=str)
    parser.add_argument("--annotations_file", default="/data/datasets/HurricaneVidNet_Dataset/output.json", type=str)
    parser.add_argument("--out_json", default="datasets/custom_videos/raw/test.json", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load your existing COCO annotations
    with open(args.annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create the format expected by UniVS
    categories = coco_data.get('categories', [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}])
    dataset = {'videos': [], 'categories': categories, 'annotations': []}
    
    # Process videos from your dataset structure
    video_dirs = [d for d in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, d))]
    print(f"Found {len(video_dirs)} video directories")
    
    for video_name in video_dirs:
        video_path = os.path.join(args.video_dir, video_name)
        image_files = [f for f in os.listdir(video_path) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            continue
            
        # Get image dimensions from the first image
        sample_img_path = os.path.join(video_path, image_files[0])
        img = cv2.imread(sample_img_path)
        if img is None:
            print(f"Warning: Could not read image {sample_img_path}")
            continue
            
        height, width = img.shape[:2]
        
        # Sort image files to ensure correct order
        image_files.sort()
        
        # Create file paths relative to the dataset root
        file_names = [os.path.join(video_name, img_file) for img_file in image_files]
        
        vid_dict = {
            "length": len(file_names),
            "file_names": file_names,
            "width": width,
            "height": height,
            "id": video_name
        }
        dataset["videos"].append(vid_dict)
    
    # Save the formatted dataset
    print(f'Total videos: {len(dataset["videos"])}')
    print(f'Saving data into {args.out_json}')
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(dataset, f)