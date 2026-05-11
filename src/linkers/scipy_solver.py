from typing import List
import scipy.optimize  # type: ignore
import scipy.sparse  # type: ignore

import numpy as np


# OLD approach, much slower. (We know in advance the number of elements in the COO matrix)
# @numba.njit
# def _build_coo_constraints(indices: np.ndarray, sizes: np.ndarray) -> Tuple[List[int], List[int]]:
#     """Return the coo format of the linear constraint matrix A

#     We find the optimal X \\in {0, 1}^n, such that C^t X is minimal
#     with AX = 1 (A enforces the selection of a track/detection once and only once in the set of hypotheses)

#     Args:
#         indices (np.ndarray): For each hypothesis, it holds the indices of the associated object in each set.
#             Shape: (n_h, n_set), dtype: int
#         sizes (np.ndarray): Size of each set
#             Shape: (n_set), dtype: int

#     Returns:
#         np.ndarray: rows of valid elements (always 1)
#         np.ndarray: cols of valid elements (always 1)
#     """
#     rows = []
#     cols = []

#     constraint_id = 0
#     for i, size in enumerate(sizes):
#         for j in range(size):
#             for k in range(len(indices)):
#                 if indices[k, i] == j:
#                     rows.append(constraint_id)
#                     cols.append(k)

#             constraint_id += 1
#     return (rows, cols)


def build_coo_constraints(indices: np.ndarray, sizes: List[int]) -> scipy.sparse.coo_array:
    """Return the coo format of the linear constraint matrix A

    We find the optimal X \\in {0, 1}^n, such that C^t X is minimal
    with AX = 1 (A enforces the selection of a track/detection once and only once in the set of hypotheses)


    Args:
        indices (np.ndarray): For each hypothesis, it holds the indices of the associated object in each set.
            Shape: (n_h, n_set), dtype: int
        sizes (List[int]): Size of each set

    Returns:
        scipy.sparse.coo_array: Sparse constraint matrix A

    """
    hypotheses_id, set_id = np.nonzero(indices != -1)
    hypotheses_id = hypotheses_id.astype(np.int32)
    set_id = set_id.astype(indices.dtype)
    objects_id = indices[hypotheses_id, set_id]

    offset = np.cumsum([0] + [size for size in sizes], dtype=indices.dtype)

    objects_uid = offset[set_id] + objects_id

    return scipy.sparse.coo_array(
        (np.ones(len(hypotheses_id), dtype=np.float32), (objects_uid, hypotheses_id)),
        shape=(offset[-1], indices.shape[0]),
    )
    # With old approach
    # rows, cols = _build_coo_constraints(indices, np.array(sizes))
    # return scipy.sparse.coo_array(
    #     (np.ones(len(rows), dtype=np.float32), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
    #     shape=(rows[-1] + 1, indices.shape[0]),
    # )


def solve_map(cost: np.ndarray, indices: np.ndarray, sizes: List[int]) -> np.ndarray:
    """Solves a Multidimensional association problem

    It assumes that only a subset of H hypotheses are kept between the sets of size `sizes`

    Args:
        cost (np.ndarray): The cost for each hypothesis
            Shape: (n_h,), dtype: float
        indices (np.ndarray): For each hypothesis, it holds the indices of the associated object in each set.
            Shape: (n_h, n_set), dtype: int
        sizes (List[int]): Size of each set

    Returns:
        np.ndarray: Selected hypotheses
            Shape: (n_h), dtype: bool

    """
    # All variables are bounded integer in {0, 1} (binary variable)
    integrality = np.full(cost.size, 3, dtype=np.int32)
    bounds = scipy.optimize.Bounds(0, 1)

    coo_constraints = build_coo_constraints(indices, sizes)
    constraints = scipy.optimize.LinearConstraint(coo_constraints, 1, 1)

    # Presolving is slow, can be set to true for debugging purposes
    res = scipy.optimize.milp(
        cost, integrality=integrality, bounds=bounds, constraints=constraints, options={"presolve": False}
    )

    if not isinstance(res.x, np.ndarray):
        raise ValueError(f"Scipy was unable to solve the MILP problem: {res.message}")

    return res.x > 0.5
