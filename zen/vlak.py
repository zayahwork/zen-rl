PRIORITY = [
    'LR_TOO_HIGH',
    'DEAD_RUN',
    'ADVANTAGE_NOT_NORMALIZED',
    'CAPPED_PERFORMANCE',
    'POLICY_COLLAPSE'
]

def rank(issues: list) -> list:
    def priority_key(mode):
        return PRIORITY.index(mode) if mode in PRIORITY else 999
    
    return sorted(issues, key=lambda issue: priority_key(issue['mode']))
