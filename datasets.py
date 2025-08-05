import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

# Class name to index mapping — update to your actual class names
CLASS_NAMES = ['faba_bean', 'weed']
CLASS_TO_IDX = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

class CustomDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transforms = transforms

        self.image_files = [
            file for file in os.listdir(images_dir)
            if file.endswith(('.jpg', '.png', '.jpeg'))
        ]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        image_path = os.path.join(self.images_dir, img_filename)
        annotation_path = os.path.join(self.annotations_dir, os.path.splitext(img_filename)[0] + ".xml")

        image = Image.open(image_path).convert("RGB")
        boxes, labels = self.parse_voc_xml(annotation_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target

    def parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            name = obj.find("name").text.lower().strip()
            if name not in CLASS_TO_IDX:
                continue  # Skip unknown classes

            label = CLASS_TO_IDX[name]
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels
