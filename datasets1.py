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
import torchvision.transforms as T
from custom_utils import collate_fn  # keep collate_fn for dataloader

# Normalization using ImageNet mean and std (ResNet50 backbone)
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])

class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = []
        
        for file_type in self.image_file_types:
            self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
        self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # Read image and convert BGR->RGB, float32
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0  # scale to [0,1]

        # Convert to tensor and normalize
        image_tensor = torch.tensor(image_resized).permute(2, 0, 1)  # HWC to CHW
        image_tensor = normalize(image_tensor)

        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        for member in root.findall('object'):
            labels.append(self.classes.index(member.find('name').text))
            
            xmin = int(member.find('bndbox').find('xmin').text)
            xmax = int(member.find('bndbox').find('xmax').text)
            ymin = int(member.find('bndbox').find('ymin').text)
            ymax = int(member.find('bndbox').find('ymax').text)
            
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            if xmax_final == xmin_final:
                xmax_final += 1
            if ymax_final == ymin_final:
                ymax_final += 1
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([idx])

        # -- COMMENTED OUT AUGMENTATION --
        # if self.transforms:
        #     sample = self.transforms(image=image_resized,
        #                              bboxes=target['boxes'],
        #                              labels=labels)
        #     image_resized = sample['image']
        #     target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)

        return image_tensor, target

    def __len__(self):
        return len(self.all_images)

def create_train_dataset(DIR):
    return CustomDataset(DIR, RESIZE_TO, RESIZE_TO, CLASSES, transforms=None)

def create_valid_dataset(DIR):
    return CustomDataset(DIR, RESIZE_TO, RESIZE_TO, CLASSES, transforms=None)

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

if __name__ == '__main__':
    dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
    print(f"Number of training images: {len(dataset)}")
