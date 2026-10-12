import numpy as np

def match(cost_matrix):
    # Minimal mock of the Hungarian algorithm used in SORT
    matches = []
    if cost_matrix.size == 0:
        return matches
    
    # Naive greedy matching for visualization tests
    flat_indices = np.argsort(cost_matrix.ravel())
    used_rows = set()
    used_cols = set()
    for idx in flat_indices:
        r, c = divmod(idx, cost_matrix.shape[1])
        if r not in used_rows and c not in used_cols:
            matches.append((r, c))
            used_rows.add(r)
            used_cols.add(c)
    return matches
