def resolve(goal, catalog):
    """
    Resolves a user goal to the most semantically relevant capability based on intent.
    Improves semantic capability reuse by incorporating synonym expansion and context-aware matching.
    """
    # Normalize input goal and capability descriptions
    goal_lower = goal.lower().strip()
    goal_terms = set(goal_lower.split())
    
    # Define synonym maps for common phrases
    synonym_map = {
        'line': {'line', 'row', 'entry', 'item', 'segment'},
        'count': {'count', 'number', 'total', 'how many', 'how much'},
        'text': {'text', 'string', 'content', 'data', 'paragraph'},
        'actual': {'real', 'true', 'non-empty'},
        'how many': {'how many', 'how much', 'what is the number of'},
        'in': {'in', 'within', 'of', 'from'},
        'content': {'content', 'material', 'information'},
        'lines': {'lines', 'rows', 'entries'}
    }
    
    # Expand goal terms using synonyms
    expanded_terms = goal_terms.copy()
    for term in goal_terms:
        if term in synonym_map:
            expanded_terms.update(synonym_map[term])
    
    best = None
    best_score = 0
    
    for capability_id, description in catalog.items():
        description_lower = description.lower().replace('.', ' ').replace(',', ' ').strip()
        description_terms = set(description_lower.split())
        
        # Expand capability description terms with synonyms
        expanded_description_terms = description_terms.copy()
        for term in description_terms:
            if term in synonym_map:
                expanded_description_terms.update(synonym_map[term])
        
        # Compute semantic overlap using expanded terms
        overlap = len(expanded_terms & expanded_description_terms)
        
        if overlap > best_score:
            best = capability_id
            best_score = overlap
    
    # Return capability if semantic match is strong (at least 2 overlapping terms)
    return best if best_score >= 2 else None