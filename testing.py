import os
import json
import cv2

def yolo_to_coco(yolo_dir, img_dir, output_file):
    coco_output = {
        "images": [],
        "annotations": [],
        "categories": [{
            "supercategory": "none",
            "id": 1,
            "name": "crack"
        }],
        "info": {},
        "licenses": []
    }

    annotation_id = 0
    category_id = 1  # For "crack"

    for idx, txt_file in enumerate(os.listdir(yolo_dir)):
        if txt_file.endswith('.txt'):
            img_file = os.path.splitext(txt_file)[0] + '.jpg'
            img_path = os.path.join(img_dir, img_file)
            
            # Load image to get dimensions
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            # Add image info to COCO (even if no annotations exist)
            image_info = {
                "file_name": img_file,
                "height": height,
                "width": width,
                "id": idx
            }
            coco_output["images"].append(image_info)

            # Read YOLO txt file
            with open(os.path.join(yolo_dir, txt_file), 'r') as f:
                lines = f.readlines()

                # Check if the file is empty (no annotations)
                if not lines:
                    # If empty, skip adding any annotations for this image
                    continue

                # Process annotations if they exist
                for line in lines:
                    label, x_center, y_center, w, h = map(float, line.strip().split())
                    
                    # Convert YOLO format (normalized) to COCO format (absolute pixel values)
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height

                    x_min = x_center - (w / 2)
                    y_min = y_center - (h / 2)

                    # Bounding box format: [x_min, y_min, width, height]
                    bbox = [x_min, y_min, w, h]
                    
                    annotation_info = {
                        "id": annotation_id,
                        "image_id": idx,
                        "category_id": category_id,
                        "bbox": bbox,
                        "area": w * h,
                        "iscrowd": 0,
                        "segmentation": []  # For detection tasks, segmentation can be left empty
                    }
                    coco_output["annotations"].append(annotation_info)
                    annotation_id += 1

    # Save the COCO formatted JSON
    with open(output_file, 'w') as out_file:
        json.dump(coco_output, out_file, indent=4)


if __name__ == "__main__":
    yolo_dir = "datasets/TrainTxt"  # Folder with .txt annotation files
    img_dir = "datasets/images/train"  # Folder with images corresponding to the annotations
    output_file = "datasets/annotations/instances_train.json"
    
    yolo_to_coco(yolo_dir, img_dir, output_file)
