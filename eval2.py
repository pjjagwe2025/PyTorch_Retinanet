import torch
from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS, TEST_DIR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader

# Evaluation function
def evaluate(test_data_loader, model):
    """Evaluate model on test data."""
    model.eval()
    
    # Initialize tqdm progress bar
    prog_bar = tqdm(test_data_loader, total=len(test_data_loader))
    target = []
    preds = []
    
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images)  # ← No targets for inference
        
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
            
        prog_bar.set_description("Evaluating on test data")
    
    # Calculate metrics
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    print("=" * 60)
    print("EVALUATING ON TEST DATA")
    print("=" * 60)
    
    # Load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    
    print(f"Loaded best model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create test dataset and loader
    test_dataset = create_valid_dataset(TEST_DIR)
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)
    
    print(f"Number of test samples: {len(test_dataset)}")
    
    # Evaluate on test data
    metric_summary = evaluate(test_loader, model)
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:      {metric_summary['map_50']*100:.3f}%")
    print(f"mAP@0.5:0.95: {metric_summary['map']*100:.3f}%")
    print(f"mAP@0.75:     {metric_summary['map_75']*100:.3f}%")
    print("=" * 60)
    
    # Additional detailed metrics
    if 'map_per_class' in metric_summary:
        print("\nPer-class mAP@0.5:0.95:")
        for i, class_map in enumerate(metric_summary['map_per_class']):
            if i < len(CLASSES) - 1:  # Skip background class
                class_name = CLASSES[i + 1] if i + 1 < len(CLASSES) else f"Class_{i+1}"
                print(f"  {class_name}: {class_map*100:.3f}%")
        print("=" * 60)