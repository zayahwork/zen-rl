RECOMMENDATIONS = {
    "LR_TOO_HIGH": "Drop learning rate to 3e-4 and add a 500 step warmup, then rerun.",
    "POLICY_COLLAPSE": "Policy entropy hit zero and stopped learning. Lower the learning rate and add an entropy bonus of 0.01.",
    "ADVANTAGE_NOT_NORMALIZED": "Set normalize_advantage=True. The run may still learn but this will break on harder environments."
}

PRIORITY = ["LR_TOO_HIGH", "ADVANTAGE_NOT_NORMALIZED", "POLICY_COLLAPSE"]

def rank(issues: list) -> list:
    return sorted(issues, key=lambda issue: PRIORITY.index(issue['mode']))

def recommend(issues: list) -> list:
    if not issues:
        return ["No failure detected. Run looks healthy."]
    
    ranked_issues = rank(issues)
    recommendations = []
    
    for issue in ranked_issues:
        mode = issue['mode']
        step = issue['step']
        value = issue['value']
        metric = issue['metric']
        
        recommendation = RECOMMENDATIONS.get(mode, f"No recommendation found for issue: {mode}")
        evidence = f"Evidence: {metric} = {value} at step {step}"
        
        recommendations.append(f"{recommendation} {evidence}")
    
    return recommendations
