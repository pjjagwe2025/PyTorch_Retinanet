from config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR
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
            # Get validation loss
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
    # Create output directory
    os.makedirs('outputs', exist_ok=True)
    
    # Create datasets and data loaders
    train_dataset = create_train_dataset(TRAIN_DIR)
    valid_dataset = create_valid_dataset(VALID_DIR)
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(model)
    
    # Total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
    # Initialize optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=15, gamma=0.1, verbose=True
    )

    # Initialize tracking variables
    train_loss_hist = Averager()
    train_loss_list = []
    val_loss_list = []
    map_50_list = []
    map_list = []

    # Model name to save with
    MODEL_NAME = 'model'

    # Visualize transformed images if requested
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)

    # Initialize best model saver and metric calculator
    save_best_model = SaveBestModel()
    metric = MeanAveragePrecision()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch
        train_loss_hist.reset()

        # Start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model, optimizer, train_loss_hist)
        metric_summary, val_loss = validate(valid_loader, model, metric)
        
        print(f"Epoch #{epoch+1} train loss: {train_loss:.3f}")   
        print(f"Epoch #{epoch+1} val loss: {val_loss:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5: {metric_summary['map_50']:.3f}")   
        print(f"Epoch #{epoch+1} mAP@0.5:0.95: {metric_summary['map']:.3f}")   
        
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch+1}")

        # Store metrics
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])

        # Save the best model
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        
        # Save the current epoch model
        save_model(epoch, model, optimizer)

        # Save plots
        save_loss_plot(OUT_DIR, train_loss_list, val_loss_list)
        save_mAP(OUT_DIR, map_50_list, map_list)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        print("-" * 50)

    print("Training completed!")

if __name__ == '__main__':
    main()