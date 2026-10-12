def match(cost_matrix):
    # Minimal mock: returns a simple greedy match
    # cost_matrix: (N, M)
    # returns: list of (row, col)
    import numpy as np
    matches = []
    rows = np.arange(cost_matrix.shape[0])
    cols = np.arange(cost_matrix.shape[1])
    
    # Very naive greedy matching for visualization purposes
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
