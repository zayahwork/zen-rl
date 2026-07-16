def detect(metrics: dict) -> list[dict]:
    """
    Detect anomalies or issues based on the provided metrics.
    
    Args:
        metrics (dict): A dictionary containing various metrics.
        
    Returns:
        list[dict]: A list of detected issues or anomalies, each represented as a dictionary with keys 'mode', 'step', 'value', and 'metric'.
    """
    issues = []
    
    # Rule 1: if any approx_kl value exceeds 5.0, add "LR_TOO_HIGH"
    if 'train/approx_kl' in metrics:
        for step, value in metrics['train/approx_kl']:
            if value > 5.0:
                issues.append({
                    'mode': "LR_TOO_HIGH",
                    'step': step,
                    'value': value,
                    'metric': 'train/approx_kl'
                })
                break
    
    # Rule 2: if any abs(entropy_loss) < 1e-6 at a step greater than 5000, add "POLICY_COLLAPSE"
    if 'train/entropy_loss' in metrics:
        for step, value in metrics['train/entropy_loss']:
            if step > 5000 and abs(value) < 1e-6:
                issues.append({
                    'mode': "POLICY_COLLAPSE",
                    'step': step,
                    'value': value,
                    'metric': 'train/entropy_loss'
                })
                break
    
    # Rule 3: if any abs(policy_gradient_loss) > 1.0, add "ADVANTAGE_NOT_NORMALIZED"
    if 'train/policy_gradient_loss' in metrics:
        for step, value in metrics['train/policy_gradient_loss']:
            if abs(value) > 1.0:
                issues.append({
                    'mode': "ADVANTAGE_NOT_NORMALIZED",
                    'step': step,
                    'value': value,
                    'metric': 'train/policy_gradient_loss'
                })
                break
    
    # Rule 4: if the final value is less than 50, add mode DEAD_RUN with the final step and value, and metric rollout/ep_rew_mean
    if 'rollout/ep_rew_mean' in metrics:
        final_step = max(step for step, _ in metrics['rollout/ep_rew_mean'])
        final_value = [value for step, value in metrics['rollout/ep_rew_mean'] if step == final_step][0]
        if final_value < 50:
            issues.append({
                'mode': "DEAD_RUN",
                'step': final_step,
                'value': final_value,
                'metric': 'rollout/ep_rew_mean'
            })
    
    # Rule 5: if the final value is between 50 and 250, add mode CAPPED_PERFORMANCE with the final step and value
    if 'rollout/ep_rew_mean' in metrics:
        final_step = max(step for step, _ in metrics['rollout/ep_rew_mean'])
        final_value = [value for step, value in metrics['rollout/ep_rew_mean'] if step == final_step][0]
        if 50 <= final_value < 250:
            issues.append({
                'mode': "CAPPED_PERFORMANCE",
                'step': final_step,
                'value': final_value,
                'metric': 'rollout/ep_rew_mean'
            })
    
    return issues
