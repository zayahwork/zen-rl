def route(metrics: dict, issues: list) -> int:
    if issues:
        return 0
    if 'rollout/ep_rew_mean' not in metrics:
        return -1  # Handle missing metric value
    max_ep_rew_mean = max(metrics.get('rollout/ep_rew_mean', []))
    current_ep_rew_mean = metrics['rollout/ep_rew_mean'][-1]
    if current_ep_rew_mean < max_ep_rew_mean / 2:
        return 2
    return 0
