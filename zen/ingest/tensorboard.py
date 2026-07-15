import os
from tbparse import SummaryReader
import pandas as pd

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
    df = reader.scalars
    
    data = {}
    
    for tag in df['tag'].unique():
        tag_df = df[df['tag'] == tag]
        values = list(zip(tag_df['step'], tag_df['value']))
        values.sort(key=lambda x: x[0])
        data[tag] = values
    
    return data
