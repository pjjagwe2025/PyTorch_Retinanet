import torch
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS, TEST_DIR, CLASSES
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

# Evaluation function
def validate(valid_data_loader, model):
    model.eval()
    
    # Initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            # FIXED: Don't pass targets during inference - only images
            outputs = model(images)
        
        # For mAP calculation using Torchmetrics
        for j in range(len(images)):  # FIXED: Use 'j' instead of 'i' to avoid conflict
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[j]['boxes'].detach().cpu()
            true_dict['labels'] = targets[j]['labels'].detach().cpu()
            
            # Add confidence threshold filtering (optional but recommended)
            confidence_threshold = 0.01  # Lower threshold to capture more predictions
            valid_indices = outputs[j]['scores'] > confidence_threshold
            
            preds_dict['boxes'] = outputs[j]['boxes'][valid_indices].detach().cpu()
            preds_dict['scores'] = outputs[j]['scores'][valid_indices].detach().cpu()
            preds_dict['labels'] = outputs[j]['labels'][valid_indices].detach().cpu()
            
            preds.append(preds_dict)
            target.append(true_dict)
        
        # Update progress bar with useful info
        total_detections = sum(len(output['boxes']) for output in outputs)
        prog_bar.set_description(f"Validating - Detections: {total_detections}")
    
    # Calculate metrics
    metric = MeanAveragePrecision(class_metrics=True)  # Enable per-class metrics
    metric.update(preds, target)
    metric_summary = metric.compute()
    
    # Debug information
    total_targets = sum(len(t['boxes']) for t in target)
    total_predictions = sum(len(p['boxes']) for p in preds)
    unique_pred_labels = set()
    unique_true_labels = set()
    
    # Collect confidence score statistics
    all_scores = []
    for p in preds:
        if len(p['labels']) > 0:
            unique_pred_labels.update(p['labels'].tolist())
            all_scores.extend(p['scores'].tolist())
    for t in target:
        if len(t['labels']) > 0:
            unique_true_labels.update(t['labels'].tolist())
    
    print(f"\nValidation completed:")
    print(f"Total ground truth boxes: {total_targets}")
    print(f"Total predictions (after filtering): {total_predictions}")
    print(f"Unique ground truth labels: {sorted(unique_true_labels)}")
    print(f"Unique predicted labels: {sorted(unique_pred_labels)}")
    
    if all_scores:
        import numpy as np
        scores_array = np.array(all_scores)
        print(f"Confidence score statistics:")
        print(f"  Mean: {scores_array.mean():.4f}")
        print(f"  Std:  {scores_array.std():.4f}")
        print(f"  Min:  {scores_array.min():.4f}")
        print(f"  Max:  {scores_array.max():.4f}")
        print(f"  Scores > 0.1: {np.sum(scores_array > 0.1)}")
        print(f"  Scores > 0.3: {np.sum(scores_array > 0.3)}")
        print(f"  Scores > 0.5: {np.sum(scores_array > 0.5)}")
    
    print(f"Available metric keys: {list(metric_summary.keys())}")
    
    return metric_summary

if __name__ == '__main__':  # FIXED: Correct syntax
    print("=" * 50)
    
    # Per-class metrics
    if 'map_per_class' in metric_summary and metric_summary['map_per_class'] is not None:
        print("\nPER-CLASS mAP@0.5:0.95:")
        print("-" * 30)
        map_per_class = metric_summary['map_per_class']
        
        # Handle both scalar and tensor cases
        if map_per_class.dim() == 0:
            # Single class or no valid detections
            if map_per_class.item() == -1:
                print("No valid detections found for mAP calculation")
            else:
                print(f"Overall: {map_per_class.item()*100:.3f}%")
        else:
            # Multiple classes - dynamically use whatever classes are available
            print("Per-class breakdown:")
            for i, class_map in enumerate(map_per_class):
                # Determine result
                if class_map.item() == -1:
                    class_result = "No valid detections"
                else:
                    class_result = f"{class_map.item()*100:.3f}%"
                
                # Skip background class (index 0) in display if it exists
                if i == 0 and len(CLASSES) > 0 and ('background' in CLASSES[0].lower() or CLASSES[0].startswith('__')):
                    continue
                    
                # Use actual class name from config, regardless of what it is
                if i < len(CLASSES):
                    class_name = CLASSES[i]
                    print(f"  {class_name:15}: {class_result}")
                else:
                    # Fallback if somehow we have more classes than names
                    print(f"  Class_{i:2d}      : {class_result}")
    else:
        print("\nPer-class metrics not available")
    
    print("\nRECOMMENDATIONS:")
    print("-" * 30)
    print("• Model needs more training epochs (try 20-50 epochs)")
    print("• Consider lower learning rate or learning rate scheduling")
    print("• Check if data augmentation is helping or hurting")
    print("• Verify bounding box coordinates are in correct format")
    print("• Consider using a pre-trained backbone if not already")
    
    print("=" * 50)
    print("VALIDATING MODEL")
    print("=" * 50)
    
    # Load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    
    print(f"Loaded model from epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Create test dataset and loader
    test_dataset = create_valid_dataset(TEST_DIR)
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Class names: {CLASSES}")
    print(f"Number of batches: {len(test_loader)}")
    
    # Run validation
    metric_summary = validate(test_loader, model)
    
    # Print results
    print("\n" + "=" * 50)
    print("VALIDATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5:      {metric_summary['map_50']*100:.3f}%")
    print(f"mAP@0.5:0.95: {metric_summary['map']*100:.3f}%")
    print(f"mAP@0.75:     {metric_summary['map_75']*100:.3f}%")
    
    # Additional metrics if available
    if 'map_small' in metric_summary:
        print(f"mAP (small):  {metric_summary['map_small']*100:.3f}%")
    if 'map_medium' in metric_summary:
        print(f"mAP (medium): {metric_summary['map_medium']*100:.3f}%")
    if 'map_large' in metric_summary:
        print(f"mAP (large):  {metric_summary['map_large']*100:.3f}%")
    
    print("=" * 50)
    
    # Debug: Check a single prediction
    print("\nSample prediction analysis:")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(test_loader))
        images, targets = sample_batch
        images = list(image.to(DEVICE) for image in images)
        sample_outputs = model(images)
        
        for idx, output in enumerate(sample_outputs):
            print(f"Sample {idx}:")
            print(f"  Predictions: {len(output['boxes'])}")
            if len(output['boxes']) > 0:
                print(f"  Score range: {output['scores'].min().item():.3f} - {output['scores'].max().item():.3f}")
                print(f"  Label range: {output['labels'].min().item()} - {output['labels'].max().item()}")
            print(f"  Ground truth boxes: {len(targets[idx]['boxes'])}")