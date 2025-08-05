import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

# The dataset class.
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        # Get all the image paths in sorted order.
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)
        
        # Set up annotations directory path
        # If dir_path is 'custom_data/train/images', we want 'custom_data/train/annotations'
        base_dir = os.path.dirname(self.dir_path)  # Get 'custom_data/train'
        self.annot_dir = os.path.join(base_dir, 'annotations')
        
        # Verify annotations directory exists
        if not os.path.exists(self.annot_dir):
            raise FileNotFoundError(f"Annotations directory not found: {self.annot_dir}")
        
        # Filter images that have corresponding annotation files
        valid_images = []
        for image_name in self.all_images:
            # Get annotation filename (change extension to .xml)
            annot_filename = os.path.splitext(image_name)[0] + '.xml'
            annot_file_path = os.path.join(self.annot_dir, annot_filename)
            
            if os.path.exists(annot_file_path):
                valid_images.append(image_name)
            else:
                print(f"Warning: No annotation found for {image_name}")
        
        self.all_images = valid_images
        print(f"Found {len(self.all_images)} images with corresponding annotations")

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Get annotation file path
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.annot_dir, annot_filename)
        
        # Verify annotation file exists (should exist due to filtering in __init__)
        if not os.path.exists(annot_file_path):
            raise FileNotFoundError(f"Annotation file not found: {annot_file_path}")
        
        boxes = []
        labels = []
        
        try:
            tree = et.parse(annot_file_path)
            root = tree.getroot()
        except Exception as e:
            raise Exception(f"Error parsing XML file {annot_file_path}: {str(e)}")
        
        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Box coordinates for xml files are extracted 
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            class_name = member.find('name').text
            if class_name not in self.classes:
                print(f"Warning: Unknown class '{class_name}' in {annot_filename}")
                continue
                
            labels.append(self.classes.index(class_name))
            
            # Get bounding box coordinates
            bndbox = member.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymin = int(float(bndbox.find('ymin').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            # Resize the bounding boxes according 
            # to resized image `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            # Check that max coordinates are at least one pixel
            # larger than min coordinates.
            if xmax_final <= xmin_final:
                xmax_final = xmin_final + 1
            if ymax_final <= ymin_final:
                ymax_final = ymin_final + 1
                
            # Check that all coordinates are within the image.
            xmin_final = max(0, xmin_final)
            ymin_final = max(0, ymin_final)
            xmax_final = min(self.width, xmax_final)
            ymax_final = min(self.height, ymax_final)
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Handle case where no valid objects found
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # Bounding box to tensor.
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # Labels to tensor.
            labels = torch.as_tensor(labels, dtype=torch.int64)
            # Area of the bounding boxes.
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # No crowd instances.
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# Prepare the final datasets and data loaders.
def create_train_dataset(DIR, use_augmentation=False):
    if use_augmentation:
        train_dataset = CustomDataset(
            DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform()
        )
    else:
        train_dataset = CustomDataset(
            DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
        )
    return train_dataset

def create_valid_dataset(DIR):
    valid_dataset = CustomDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return valid_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader


# execute `datasets.py`` using Python command from 
# Terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        if len(target['boxes']) == 0:
            print("No bounding boxes found in this image")
            return
            
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 0, 255), 
                2
            )
            cv2.putText(
                image, 
                label, 
                (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(min(NUM_SAMPLES_TO_VISUALIZE, len(dataset))):
        image, target = dataset[i]
        visualize_sample(image, target)