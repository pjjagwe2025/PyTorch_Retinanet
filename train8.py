import argparse
import os
import time
import gc
import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNetClassificationHead, Retinanet_ResNet50_FPN_Weights

from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,
    RESIZE_TO, VALID_DIR, TRAIN_DIR, BATCH_SIZE, CLASSES
)
from datasets import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot, save_mAP

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)


# =================== Early Stopping ===================
class EarlyStopping:
    """Stop training when validation mAP stops improving."""
    def __init__(self, patience=7, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_map = 0
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_map, model):
        if val_map < self.best_map + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        else:
            self.best_map = val_map
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False


# =================== Model Creation ===================
def create_model(num_classes, pretrained=True, freeze_backbone=False):
    """Create RetinaNet with custom number of classes and optional frozen backbone."""
    weights = Retinanet_ResNet50_FPN_Weights.DEFAULT if pretrained else None
    model = retinanet_resnet50_fpn(weights=weights, weights_backbone=weights)

    # Replace classification head
    in_features = model.head.classification_head.conv[0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_features,
        num_anchors=num_anchors,
        num_classes=num_classes
    )

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("🔒 Backbone frozen.")

    return model


# =================== Optimizer ===================
def get_optimizer(name, parameters, lr, weight_decay=0.0):
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


# =================== Argument Parser ===================
def parse_arguments():
    parser = argparse.ArgumentParser(description='RetinaNet Training Script')

    # Training
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam','adamw','sgd','rmsprop'])
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001)
    parser.add_argument('--scheduler_step_size', type=int, default=15)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--no_scheduler', action='store_true')

    # Data
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--resize_to', type=int, default=RESIZE_TO)
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--valid_dir', type=str, default=VALID_DIR)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--resume', type=str, default=None)

    # Backbone options
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained backbone')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')

    return parser.parse_args()


# =================== Training Loop ===================
def train_one_epoch(train_loader, model, optimizer):
    model.train()
    prog_bar = tqdm(train_loader, total=len(train_loader))
    train_loss_hist = Averager()

    for images, targets in prog_bar:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k,v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        train_loss_hist.send(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        prog_bar.set_description(f"Loss: {loss.item():.4f}")

    return train_loss_hist.value


# =================== Validation ===================
def validate(valid_loader, model, class_names, score_threshold=0.05):
    metric = MeanAveragePrecision(
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True,
        max_detection_thresholds=[100, 300, 500],
        extended_summary=True
    ).to(DEVICE)

    model.eval()
    preds, targets = [], []

    with torch.no_grad():
        for images, targs in tqdm(valid_loader, desc="Validating", leave=False):
            images = [img.to(DEVICE) for img in images]
            targs = [{k: v.to(DEVICE) for k, v in tar.items()} for tar in targs]

            outputs = model(images)
            filtered_outputs = []
            for out in outputs:
                keep = out['scores'] > score_threshold
                filtered_outputs.append({
                    'boxes': out['boxes'][keep],
                    'scores': out['scores'][keep],
                    'labels': out['labels'][keep]
                })
            preds.extend(filtered_outputs)
            targets.extend(targs)

    metric.update(preds, targets)
    results = metric.compute()

    map50 = results['map'].item()
    precision = results['precision']
    recall = results['recall']
    map_per_class = results['map_per_class']

    # Print per-class metrics
    print("=== Per-class Metrics ===")
    for i, cname in enumerate(class_names):
        print(f"{cname}: mAP@0.5={map_per_class[i]:.4f} | Precision={precision[i]:.4f} | Recall={recall[i]:.4f}")

    return results


# =================== Main ===================
def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)

    # Datasets & loaders
    train_dataset = create_train_dataset(args.train_dir)
    valid_dataset = create_valid_dataset(args.valid_dir)
    train_loader = create_train_loader(train_dataset, args.num_workers)
    valid_loader = create_valid_loader(valid_dataset, args.num_workers)

    # Model
    model = create_model(
        num_classes=NUM_CLASSES,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone
    ).to(DEVICE)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])

    # Optimizer & scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, params, args.lr, args.weight_decay)
    scheduler = None
    if not args.no_scheduler:
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)

    # Early stopping
    early_stopping = None
    if args.early_stopping_patience is not None:
        early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta)

    # Metrics CSV
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        pd.DataFrame(columns=["epoch","class_name","map50","precision","recall","lr","loss"]).to_csv(csv_path, index=False)

    best_map50 = 0.0
    save_best_model = SaveBestModel()

    # Training loop
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        train_loss = train_one_epoch(train_loader, model, optimizer)
        results = validate(valid_loader, model, CLASSES)

        # Average metrics
        map50 = results['map'].item()
        overall_precision = results['precision'].mean().item()
        overall_recall = results['recall'].mean().item()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch #{epoch+1} Loss: {train_loss:.4f} | mAP@0.5: {map50:.4f} | Precision: {overall_precision:.4f} | Recall: {overall_recall:.4f}")

        # Save per-class metrics
        map_per_class = results['map_per_class']
        metrics_to_save = []
        for i, cname in enumerate(CLASSES):
            metrics_to_save.append({
                "epoch": epoch+1,
                "class_name": cname,
                "map50": map_per_class[i].item(),
                "precision": results['precision'][i].item(),
                "recall": results['recall'][i].item(),
                "lr": current_lr,
                "loss": train_loss
            })
        pd.DataFrame(metrics_to_save).to_csv(csv_path, mode='a', index=False, header=False)

        # Save best model
        if map50 > best_map50:
            best_map50 = map50
            save_best_model(model, best_map50, epoch, args.out_dir)

        # Save last model
        save_model(epoch, model, optimizer)

        # Scheduler step
        if scheduler:
            scheduler.step()

        # Early stopping
        if early_stopping and early_stopping(map50, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        # Clean up
        torch.cuda.empty_cache()
        gc.collect()

    print(f"\n✅ Training complete! Best mAP@0.5: {best_map50:.4f}")


if __name__ == '__main__':
    main()
