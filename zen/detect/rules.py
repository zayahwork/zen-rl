def detect(metrics: dict) -> list[str]:
    """
    Detect anomalies or issues based on the provided metrics.
    
    Args:
        metrics (dict): A dictionary containing various metrics.
        
    Returns:
        list[str]: A list of detected issues or anomalies.
    """
    issues = []
    
    # Rule 1: if any approx_kl value exceeds 1.0, add "LR_TOO_HIGH"
    if 'train/approx_kl' in metrics:
        for step, value in metrics['train/approx_kl']:
            if value > 1.0:
                issues.append("LR_TOO_HIGH")
    
    # Rule 2: if any abs(entropy_loss) < 1e-6 at a step greater than 5000, add "POLICY_COLLAPSE"
    if 'train/entropy_loss' in metrics:
        for step, value in metrics['train/entropy_loss']:
            if step > 5000 and abs(value) < 1e-6:
                issues.append("POLICY_COLLAPSE")
    
    # Rule 3: if any abs(policy_gradient_loss) > 1.0, add "ADVANTAGE_NOT_NORMALIZED"
    if 'train/policy_gradient_loss' in metrics:
        for step, value in metrics['train/policy_gradient_loss']:
            if abs(value) > 1.0:
                issues.append("ADVANTAGE_NOT_NORMALIZED")
    
    return issues
