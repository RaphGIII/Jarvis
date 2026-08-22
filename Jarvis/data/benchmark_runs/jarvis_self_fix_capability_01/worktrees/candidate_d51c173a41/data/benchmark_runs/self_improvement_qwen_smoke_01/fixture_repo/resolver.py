def resolve(goal, catalog):
    terms = set(goal.lower().split())
    best = None
    best_score = 0
    for capability_id, description in catalog.items():
        target = set((capability_id + " " + description).lower().replace(".", " ").split())
        score = len(terms & target)
        if score > best_score:
            best = capability_id
            best_score = score
    return best if best_score >= 2 else None
