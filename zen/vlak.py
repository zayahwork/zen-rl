RECOMMENDATIONS = {
    "LR_TOO_HIGH": "Drop learning rate to 3e-4 and add a 500 step warmup, then rerun.",
    "POLICY_COLLAPSE": "Policy entropy hit zero and stopped learning. Lower the learning rate and add an entropy bonus of 0.01.",
    "ADVANTAGE_NOT_NORMALIZED": "Set normalize_advantage=True. The run may still learn but this will break on harder environments."
}

def recommend(issues: list) -> list:
    if not issues:
        return ["No failure detected. Run looks healthy."]
    return [RECOMMENDATIONS.get(issue, f"No recommendation found for issue: {issue}") for issue in issues]
