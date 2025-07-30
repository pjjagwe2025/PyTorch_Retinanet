import argparse
from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR,
    BATCH_SIZE,
    CLASSES
)
from model import create_model
from custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import torch
import matplotlib.pyplot as plt
import time
import os
import gc

plt.style.use('ggplot')

# Set seeds for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

class EarlyStopping:
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
        raise ValueError(f"Unsupported optimizer: {name}. Choose from 'adam', 'adamw', 'sgd', 'rmsprop'")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Detection Training Script')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['adam', 'adamw', 'sgd', 'rmsprop'])
    parser.add_argument('--early_stopping_patience', type=int, default=None)
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001)
    parser.add_argument('--scheduler_step_size', type=int, default=15)
    parser.add_argument('--scheduler_gamma', type=float, default=0.1)
    parser.add_argument('--no_scheduler', action='store_true')
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--resize_to', type=int, default=RESIZE_TO)
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR)
    parser.add_argument('--valid_dir', type=str, default=VALID_DIR)
    parser.add_argument('--out_dir', type=str, default=OUT_DIR)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()

def train(train_data_loader, model, optimizer, train_loss_hist):
    print('Training')
    model.train()
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_hist.value

def validate(valid_data_loader, model, metric):
    print('Validating')
    model.eval()
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    val_loss_hist = Averager()
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_hist.send(losses.item())
            model.eval()
            outputs = model(images)
        for j in range(len(images)):
            true_dict = {'boxes': targets[j]['boxes'].cpu(), 'labels': targets[j]['labels'].cpu()}
            pred_dict = {
                'boxes': outputs[j]['boxes'].cpu(),
                'scores': outputs[j]['scores'].cpu(),
                'labels': outputs[j]['labels'].cpu()
            }
            preds.append(pred_dict)
            target.append(true_dict)
        prog_bar.set_description(desc=f"Val Loss: {val_loss_hist.value:.4f}")
    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    torch.cuda.empty_cache()
    gc.collect()
    return metric_summary, val_loss_hist.value

def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    import config
    original_batch_size = config.BATCH_SIZE
    original_resize_to = config.RESIZE_TO
    config.BATCH_SIZE = args.batch_size
    config.RESIZE_TO = args.resize_to
    train_dataset = create_train_dataset(args.train_dir)
    valid_dataset = create_valid_dataset(args.valid_dir)
    train_loader = create_train_loader(train_dataset, args.num_workers)
    valid_loader = create_valid_loader(valid_dataset, args.num_workers)
    config.BATCH_SIZE = original_batch_size
    config.RESIZE_TO = original_resize_to
    model = create_model(num_classes=NUM_CLASSES).to(DEVICE)
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, params, args.lr, args.weight_decay)
    scheduler = None
    if not args.no_scheduler:
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    train_loss_hist = Averager()
    train_loss_list = []
    val_loss_list = []
    map_50_list = []
    map_list = []
    if args.visualize or VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)
    save_best_model = SaveBestModel()
    metric = MeanAveragePrecision()
    early_stopping = EarlyStopping(args.early_stopping_patience, args.early_stopping_min_delta) if args.early_stopping_patience else None
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEPOCH {epoch+1} of {args.epochs}")
        train_loss_hist.reset()
        start_time = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_hist)
        metric_summary, val_loss = validate(valid_loader, model, metric)
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")   
        print(f"Epoch #{epoch+1} val loss: {val_loss:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5: {metric_summary['map_50']:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5:0.95: {metric_summary['map']:.3f}")
        if 'map_per_class' in metric_summary and 'map_50_per_class' in metric_summary:
            print("\nPer-class mAP:")
            for idx, class_name in enumerate(CLASSES):
                m95 = metric_summary['map_per_class'][idx].item()
                m50 = metric_summary['map_50_per_class'][idx].item()
                print(f"  - {class_name:>10}: mAP@0.5:0.95 = {m95:.4f}, mAP@0.5 = {m50:.4f}")
        end_time = time.time()
        print(f"Took {((end_time - start_time) / 60):.3f} minutes for epoch {epoch+1}")
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        save_best_model(model, float(metric_summary['map']), epoch, args.out_dir)
        save_model(epoch, model, optimizer)
        save_loss_plot(args.out_dir, train_loss_list, val_loss_list, save_name='loss')
        save_mAP(args.out_dir, map_50_list, map_list)
        if scheduler:
            scheduler.step()
        if early_stopping and early_stopping(metric_summary['map'], model):
            print(f"\nEarly stopping triggered after {epoch+1} epochs!")
            break
        print("-" * 50)
    print("Training completed!")
    if train_loss_list and val_loss_list and map_list:
        print(f"\nFinal Training Summary:")
        print(f"Final train loss: {train_loss_list[-1]:.4f}")
        print(f"Final validation loss: {val_loss_list[-1]:.4f}")
        print(f"Best mAP@0.5:0.95: {max(map_list):.4f}")
        print(f"Best mAP@0.5: {max(map_50_list):.4f}")

if __name__ == '__main__':
    main()
