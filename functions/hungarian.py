from numba import njit, prange
import numpy as np

@njit(cache=True, fastmath=True)
def _init_distances(cost_row, u_i, v, dist, pred, i):
    """Vectorizable: initialize distances from row i."""
    for j in range(len(v)):
        d = cost_row[j] - u_i - v[j]
        if d < dist[j]:
            dist[j] = d
            pred[j] = i

@njit(cache=True, fastmath=True)
def _relax(cost_row, u_i2, v, dist, pred, i2, min_d, visited):
    """Vectorizable: relax all unvisited columns through row i2."""
    for j in range(len(v)):
        if not visited[j]:
            d = min_d + cost_row[j] - u_i2 - v[j]
            if d < dist[j]:
                dist[j] = d
                pred[j] = i2

@njit(cache=True, fastmath=True)
def _find_min_unvisited(dist, visited, n_cols):
    """Find unvisited column with minimum distance."""
    min_d = np.float64(1e300)
    min_j = -1
    for j in range(n_cols):
        if not visited[j] and dist[j] < min_d:
            min_d = dist[j]
            min_j = j
    return min_j, min_d

@njit(cache=True, fastmath=True)
def _update_potentials(dist, visited, v, last_dist, n_cols):
    """Vectorizable: update column potentials for all visited columns."""
    for j in range(n_cols):
        if visited[j]:
            v[j] += dist[j] - last_dist

@njit(cache=True, fastmath=True)
def _lapjv(cost):
    """
    Jonker-Volgenant with forced fastmath vectorization on all inner loops.
    cost: (n_rows, n_cols), n_cols >= n_rows, dtype float64 contiguous.
    """
    n_rows, n_cols = cost.shape
    INF = np.float64(1e300)

    u       = np.zeros(n_rows, dtype=np.float64)
    v       = np.zeros(n_cols, dtype=np.float64)
    row_of_col = np.full(n_cols, -1, dtype=np.int64)
    col_of_row = np.full(n_rows, -1, dtype=np.int64)

    # --- column reduction ---
    for j in range(n_cols):
        best_i   = 0
        best_val = cost[0, j]
        for i in range(1, n_rows):
            c = cost[i, j]
            if c < best_val:
                best_val = c
                best_i   = i
        v[j] = best_val
        if col_of_row[best_i] == -1:
            col_of_row[best_i] = j
            row_of_col[j]      = best_i

    # --- row reduction ---
    for i in range(n_rows):
        best_j   = -1
        best_val = INF
        second   = INF
        for j in range(n_cols):
            red = cost[i, j] - v[j]
            if red < best_val:
                second   = best_val
                best_val = red
                best_j   = j
            elif red < second:
                second = red
        u[i] = best_val
        if best_j >= 0 and row_of_col[best_j] == -1:
            row_of_col[best_j] = i
            col_of_row[i]      = best_j
        elif best_j >= 0 and best_val < second:
            prev_i = row_of_col[best_j]
            if prev_i >= 0 and col_of_row[prev_i] == best_j:
                col_of_row[prev_i] = -1

    # --- augmenting paths ---
    dist    = np.full(n_cols, INF,   dtype=np.float64)
    pred    = np.full(n_cols, -1,    dtype=np.int64)
    visited = np.zeros(n_cols,       dtype=np.bool_)

    for i in range(n_rows):
        if col_of_row[i] >= 0:
            continue

        # reset
        for j in range(n_cols):
            dist[j]    = INF
            pred[j]    = -1
            visited[j] = False

        # vectorized distance init
        _init_distances(cost[i], u[i], v, dist, pred, i)

        last_j = -1
        while True:
            min_j, min_d = _find_min_unvisited(dist, visited, n_cols)
            if min_j == -1:
                break
            visited[min_j] = True

            if row_of_col[min_j] == -1:
                last_j = min_j
                break

            # vectorized relaxation
            i2 = row_of_col[min_j]
            _relax(cost[i2], u[i2], v, dist, pred, i2, min_d, visited)

        if last_j == -1:
            continue

        # path traceback
        j = last_j
        while j != -1:
            i2    = pred[j]
            old_j = col_of_row[i2] if 0 <= i2 < n_rows else -1
            row_of_col[j] = i2
            if 0 <= i2 < n_rows:
                col_of_row[i2] = j
            j = old_j

        # vectorized potential update
        _update_potentials(dist, visited, v, dist[last_j], n_cols)
        u[i] += dist[last_j]

    # collect results
    row_ind = np.empty(n_rows, dtype=np.int64)
    col_ind = np.empty(n_rows, dtype=np.int64)
    count   = np.int64(0)
    for i in range(n_rows):
        if col_of_row[i] >= 0:
            row_ind[count] = i
            col_ind[count] = col_of_row[i]
            count += 1

    return row_ind[:count], col_ind[:count]