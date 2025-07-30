# Add this to your datasets.py file or update existing functions

def create_train_loader(train_dataset, num_workers, batch_size=None):
    """Create training data loader with configurable batch size."""
    from config import BATCH_SIZE
    from custom_utils import collate_fn
    
    if batch_size is None:
        batch_size = BATCH_SIZE
        
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers, batch_size=None):
    """Create validation data loader with configurable batch size."""
    from config import BATCH_SIZE
    from custom_utils import collate_fn
    
    if batch_size is None:
        batch_size = BATCH_SIZE
        
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader