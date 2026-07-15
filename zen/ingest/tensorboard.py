import os
from tbparse import SummaryReader

def load_run(path: str) -> dict:
    """
    Load a TensorBoard log directory and return a dictionary mapping tag names to lists of (step, value) tuples.
    
    Args:
        path (str): The path to the TensorBoard log directory.
        
    Returns:
        dict: A dictionary mapping tag names to lists of (step, value) tuples.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Directory {path} does not exist")
    
    reader = SummaryReader(path)
    data = {}
    
    for tag in reader.tags:
        values = []
        for step, value in reader.get_tensorboard_dict(tag).items():
            values.append((step, value))
        data[tag] = values
    
    return data
