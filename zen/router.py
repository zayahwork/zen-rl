def route(metrics: dict, issues: list) -> int:
    if issues:
        return 0
    if 'rollout/ep_rew_mean' not in metrics:
        return -1  # Handle missing metric value
    
    values = [v for step, v in metrics['rollout/ep_rew_mean']]
    
    max_ep_rew_mean = max(values)
    current_ep_rew_mean = values[-1]
    
    if current_ep_rew_mean < max_ep_rew_mean / 2:
        return 2
    return 0
