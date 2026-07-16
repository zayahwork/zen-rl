RECOMMENDATIONS = {
    "LR_TOO_HIGH": "Drop learning rate to 3e-4 and add a 500 step warmup, then rerun.",
    "POLICY_COLLAPSE": "Policy entropy hit zero and stopped learning. Lower the learning rate and add an entropy bonus of 0.01.",
    "ADVANTAGE_NOT_NORMALIZED": "Set normalize_advantage=True. The run may still learn but this will break on harder environments.",
    "DEAD_RUN": "Run never learned. Reward stayed near starting value. Check learning rate is not too high or too low, and check clip_range is not tiny.",
    "CAPPED_PERFORMANCE": "Run learned but plateaued well below what is achievable. Check ent_coef is not too high, n_epochs is not too low, and gae_lambda is not zero."
}

PRIORITY = ["LR_TOO_HIGH", "DEAD_RUN", "ADVANTAGE_NOT_NORMALIZED", "CAPPED_PERFORMANCE", "POLICY_COLLAPSE"]

CAUSED_BY = {
    "POLICY_COLLAPSE": "LR_TOO_HIGH",
    "DEAD_RUN": "LR_TOO_HIGH",
}

def group(issues: list) -> list:
    modes = {i['mode'] for i in issues}
    return [i for i in issues if CAUSED_BY.get(i['mode']) not in modes]

def rank(issues: list) -> list:
    return sorted(issues, key=lambda issue: PRIORITY.index(issue['mode']))

def recommend(issues: list) -> list:
    if not issues:
        return ["No failure detected. Run looks healthy."]
    
    ranked_issues = group(rank(issues))
    recommendations = []
    
    for issue in ranked_issues:
        mode = issue['mode']
        step = issue['step']
        value = issue['value']
        metric = issue['metric']
        
        recommendation = RECOMMENDATIONS.get(mode, f"No recommendation found for issue: {mode}")
        evidence = f"Evidence: {metric} = {value:.4g} at step {step}"
        
        recommendations.append(f"{recommendation} {evidence}")
    
    return recommendations
