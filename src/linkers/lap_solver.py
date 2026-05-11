from typing import List
import scipy.optimize  # type: ignore
import scipy.sparse  # type: ignore

import numpy as np
import pylapy


def convert_to_sparse_cost(cost: np.ndarray, indices: np.ndarray, sizes: List[int]) -> scipy.sparse.coo_array:
    """Convert the 2D MAP problem to a LAP one.

    Args:
        cost (np.ndarray): Cost for each hypothesis. We assume that it is organized as
            (linking_costs, missed_costs, new_tracks_cost), such that n_h = n_link + n_track + n_det
            Shape: (n_h,), dtype: float
        indices (np.ndarray): For each hypothesis, it holds the indices of the associated object in each set.
            Shape: (n_h, 2), dtype: int
        sizes (List[int]): Size of each set (should be of length 2)

    Returns:
        scipy.sparse.coo_array: The equivalent squared cost matrix
            Shape: (n_track + n_det, n_det + n_track)

    """
    n_link = len(cost) - sum(sizes)
    n_track = sizes[0]
    n_det = sizes[1]

    # Safety checks
    assert (cost > 0).all()
    assert len(sizes) == 2, "LAP can only be used for a tree-depth of 2"
    assert (indices[-n_det:, 0] == -1).all(), "The last hypotheses should be created tracks"
    assert (indices[n_link : n_link + n_track, 1] == -1).all(), "These hypotheses should be missing detections ones"

    rows = indices[:, 0].copy()
    cols = indices[:, 1].copy()

    rows[-n_det:] = n_track + np.arange(n_det)
    cols[n_link : n_link + n_track] = n_det + np.arange(n_track)

    # Add 0.0 cost in the diagonal block to have feasible solutions
    rows = np.concatenate((rows, cols[:n_link] + n_track))
    cols = np.concatenate((cols, rows[:n_link] + n_det))
    cost = np.concatenate((cost, np.zeros(n_link, dtype=cost.dtype)))

    return scipy.sparse.coo_array((cost + 0.1, (rows, cols)), shape=(sum(sizes), sum(sizes)))


def solve_map(cost: np.ndarray, indices: np.ndarray, sizes: List[int]) -> np.ndarray:
    """Solves a linear association problem from a 2D MAP (equivalent)

    It assumes that only a subset of H hypotheses are kept between the sets of size `sizes`

    The number of set should be equal to 2, allowing to use an optimal algorithm to find the solution.
    It will use a sparse implementation of Jonker-Volgenant algorithm.

    Args:
        cost (np.ndarray): Cost for each hypothesis. We assume that it is organized as
            (linking_costs, missed_costs, new_tracks_cost), such that n_h = n_link + n_track + n_det
            Shape: (n_h,), dtype: float
        indices (np.ndarray): For each hypothesis, it holds the indices of the associated object in each set.
            Shape: (n_h, 2), dtype: int
        sizes (List[int]): Size of each set (should be of length 2)

    Returns:
        np.ndarray: Selected hypotheses
            Shape: (n_h), dtype: bool

    """

    indices = indices.copy()  # Copy as convert will modify inplace indices
    cost = cost.copy()
    coo = convert_to_sparse_cost(cost, indices, sizes)

    links = pylapy.LapSolver().sparse_solve(coo, coo.data.max(), hard=True, feasible=True)

    # Convert to selected hypotheses
    i_to_j = np.full((sum(sizes),), -1)
    i_to_j[links[:, 0]] = links[:, 1]

    return i_to_j[coo.row[: len(cost)]] == coo.col[: len(cost)]
