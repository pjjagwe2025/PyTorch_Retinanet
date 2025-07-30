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
    """Early stopping to stop training when validation mAP stops improving."""
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
    """Get optimizer based on name."""
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Object Detection Training Script')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help=f'Number of epochs to train (default: {NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='Weight decay (default: 0.0001)')
    parser.add_argument('--optimizer', type=str, default='sgd', 
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer to use (default: sgd)')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (default: None - no early stopping)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0001,
                       help='Minimum delta for early stopping (default: 0.0001)')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler_step_size', type=int, default=15,
                       help='Step size for learning rate scheduler (default: 15)')
    parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                       help='Gamma for learning rate scheduler (default: 0.1)')
    parser.add_argument('--no_scheduler', action='store_true',
                       help='Disable learning rate scheduler')
    
    # Data parameters
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS,
                       help=f'Number of data loading workers (default: {NUM_WORKERS})')
    parser.add_argument('--resize_to', type=int, default=RESIZE_TO,
                       help=f'Resize images to this size (default: {RESIZE_TO})')
    
    # Paths
    parser.add_argument('--train_dir', type=str, default=TRAIN_DIR,
                       help=f'Training data directory (default: {TRAIN_DIR})')
    parser.add_argument('--valid_dir', type=str, default=VALID_DIR,
                       help=f'Validation data directory (default: {VALID_DIR})')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR,
                       help=f'Output directory (default: {OUT_DIR})')
    
    # Other options
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize transformed images')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    return parser.parse_args()

# Function for running training iterations.
def train(train_data_loader, model, optimizer, train_loss_hist):
    print('Training')
    model.train()
    
    # Initialize tqdm progress bar
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
    
        # Update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    
    return train_loss_hist.value

# Function for running validation iterations.
def validate(valid_data_loader, model, metric):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    val_loss_hist = Averager()
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            # Get validation loss (model in training mode for loss calculation)
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss_hist.send(losses.item())
            
            # Switch to evaluation mode for inference
            model.eval()
            outputs = model(images)

        # For mAP calculation using Torchmetrics
        for j in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[j]['boxes'].detach().cpu()
            true_dict['labels'] = targets[j]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[j]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[j]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[j]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        
        # Update progress bar
        prog_bar.set_description(desc=f"Val Loss: {val_loss_hist.value:.4f}")

    # Calculate mAP
    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return metric_summary, val_loss_hist.value

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Print training configuration
    print("=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Resize To: {args.resize_to}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print(f"Device: {DEVICE}")
    print(f"Classes: {CLASSES}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Temporarily update config values if different from defaults
    import config
    original_batch_size = config.BATCH_SIZE
    original_resize_to = config.RESIZE_TO
    
    # Update config values temporarily
    if args.batch_size != BATCH_SIZE:
        config.BATCH_SIZE = args.batch_size
    if args.resize_to != RESIZE_TO:
        config.RESIZE_TO = args.resize_to
    
    # Create datasets and data loaders (they will use updated config values)
    train_dataset = create_train_dataset(args.train_dir)
    valid_dataset = create_valid_dataset(args.valid_dir)
    train_loader = create_train_loader(train_dataset, args.num_workers)
    valid_loader = create_valid_loader(valid_dataset, args.num_workers)
    
    # Restore original config values
    config.BATCH_SIZE = original_batch_size
    config.RESIZE_TO = original_resize_to
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume:
        print(f"Resuming training from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    
    print(model)
    
    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Initialize optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = get_optimizer(args.optimizer, params, args.lr, args.weight_decay)
    
    # Initialize scheduler
    scheduler = None
    if not args.no_scheduler:
        scheduler = StepLR(
            optimizer=optimizer, 
            step_size=args.scheduler_step_size, 
            gamma=args.scheduler_gamma, 
            verbose=True
        )

    # Initialize tracking variables
    train_loss_hist = Averager()
    train_loss_list = []
    val_loss_list = []
    map_50_list = []
    map_list = []

    # Visualize transformed images if requested
    if args.visualize or VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # Initialize best model saver, metric calculator, and early stopping
    save_best_model = SaveBestModel()
    metric = MeanAveragePrecision()
    early_stopping = None
    if args.early_stopping_patience is not None:
        early_stopping = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta
        )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEPOCH {epoch+1} of {args.epochs}")

        # Reset the training loss histories for the current epoch
        train_loss_hist.reset()

        # Start timer and carry out training and validation
        start_time = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_hist)
        metric_summary, val_loss = validate(valid_loader, model, metric)
        
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")   
        print(f"Epoch #{epoch+1} val loss: {val_loss:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5: {metric_summary['map_50']:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5:0.95: {metric_summary['map']:.3f}")   
        
        end_time = time.time()
        print(f"Took {((end_time - start_time) / 60):.3f} minutes for epoch {epoch+1}")

        # Store metrics
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # Save the best model
        save_best_model(
            model, float(metric_summary['map']), epoch, args.out_dir
        )
        
        # Save the current epoch model
        save_model(epoch, model, optimizer)

        # Save plots with both training and validation losses
        save_loss_plot(args.out_dir, train_loss_list, val_loss_list, save_name='loss')
        save_mAP(args.out_dir, map_50_list, map_list)
        
        # Step the learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Check early stopping
        if early_stopping is not None:
            if early_stopping(metric_summary['map'], model):
                print(f"\nEarly stopping triggered after {epoch+1} epochs!")
                break
        
        print("-" * 50)

    print("Training completed!")
    
    # Print final training summary
    if train_loss_list and val_loss_list and map_list:
        print(f"\nFinal Training Summary:")
        print(f"Final train loss: {train_loss_list[-1]:.4f}")
        print(f"Final validation loss: {val_loss_list[-1]:.4f}")
        print(f"Best mAP@0.5:0.95: {max(map_list):.4f}")
        print(f"Best mAP@0.5: {max(map_50_list):.4f}")

if __name__ == '__main__':
    main()