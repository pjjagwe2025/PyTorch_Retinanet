import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import supervision as sv

# Load the Supervision dataset
DATA_DIR = "custom_data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "valid")

train_dataset = sv.DetectionDataset.from_yolo(folder_path=TRAIN_DIR, data_yaml_path=os.path.join(DATA_DIR, "data.yaml"))
val_dataset = sv.DetectionDataset.from_yolo(folder_path=VAL_DIR, data_yaml_path=os.path.join(DATA_DIR, "data.yaml"))

# Class labels
CLASS_NAMES = train_dataset.classes
id2label = {i: name for i, name in enumerate(CLASS_NAMES)}

# Define dataset wrapper
class CustomCOCODataset(torch.utils.data.Dataset):
    def __init__(self, sv_dataset, transforms=None):
        self.dataset = sv_dataset
        self.transforms = transforms

    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        # Convert boxes to [x, y, w, h] -> [x_min, y_min, x_max, y_max]
        boxes = torch.tensor(target['bbox'], dtype=torch.float32)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x + w -> x_max
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y + h -> y_max

        target_converted = {
            'boxes': boxes,
            'labels': torch.tensor(target['class_id'], dtype=torch.int64)
        }

        if self.transforms:
            image = self.transforms(image)

        return image, target_converted

    def __len__(self):
        return len(self.dataset)

# Transforms
transform = T.Compose([
    T.ToTensor(),
])

train_dataset = CustomCOCODataset(train_dataset, transforms=transform)
val_dataset = CustomCOCODataset(val_dataset, transforms=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Load RetinaNet model
model = retinanet_resnet50_fpn_v2(pretrained=True)
num_classes = len(CLASS_NAMES)

# Replace the head with new head for our number of classes
in_features = model.head.classification_head.num_anchors * model.head.classification_head.conv[0].out_channels
model.head.classification_head.num_classes = num_classes
model.head.classification_head.cls_logits = nn.Conv2d(
    in_channels=model.head.classification_head.conv[0].out_channels,
    out_channels=model.head.classification_head.num_anchors * num_classes,
    kernel_size=3,
    stride=1,
    padding=1
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Metric
metric = MeanAveragePrecision()

# Training loop
num_epochs = 2
best_map = 0.0

for epoch in range(num_epochs):
    start = time.time()
    print(f"\nEPOCH {epoch+1} of {num_epochs}")

    # Training
    model.train()
    train_loss = 0.0
    print("Training")
    for images, targets in tqdm(train_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

    epoch_train_loss = train_loss / len(train_loader)

    # Validation
    print("Validating")
    model.eval()
    val_loss = 0.0
    metric.reset()
    with torch.no_grad():
        for images, targets in tqdm(val_loader):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

            outputs = model(images)
            outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
            targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            metric.update(outputs, targets)

    epoch_val_loss = val_loss / len(val_loader)
    metric_output = metric.compute()

    # Print global metrics
    print(f"Epoch #{epoch+1} train loss: {epoch_train_loss:.3f}")
    print(f"Epoch #{epoch+1} val loss: {epoch_val_loss:.3f}")
    print(f"Epoch #{epoch+1} mAP@0.5: {metric_output['map_50']:.3f}")
    print(f"Epoch #{epoch+1} mAP@0.5:0.95: {metric_output['map']:.3f}")

    # Print per-class mAP
    print("Per-class mAP@0.5:")
    for i, ap in enumerate(metric_output['map_50_per_class']):
        class_name = id2label.get(i, str(i))
        print(f"  Class '{class_name}': mAP = {ap:.4f}")

    # Save best model
    if metric_output['map'] > best_map:
        best_map = metric_output['map']
        torch.save(model.state_dict(), "best_retinanet_model.pth")
        print(f"\nSAVED BEST MODEL FOR EPOCH: {epoch+1}")

    end = time.time()
    print(f"Took {(end - start)/60:.3f} minutes for epoch {epoch+1}")

print("\nTraining completed!")
print(f"\nFinal Training Summary:")
print(f"Final train loss: {epoch_train_loss:.4f}")
print(f"Final validation loss: {epoch_val_loss:.4f}")
print(f"Best mAP@0.5:0.95: {best_map:.4f}")
print(f"Best mAP@0.5: {metric_output['map_50']:.4f}")
