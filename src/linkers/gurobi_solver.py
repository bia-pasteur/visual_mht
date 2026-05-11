from typing import List

import gurobipy as gp
from gurobipy import GRB  # pylint: disable=no-name-in-module
import numpy as np

from .scipy_solver import build_coo_constraints


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
    model = gp.Model("MILP_2D")  # pylint: disable=no-member

    # All variables are bounded integer in {0, 1} (binary variable)
    variables = model.addVars(cost.size, vtype=GRB.BINARY, name="X")

    # Objective
    model.setObjective(
        gp.quicksum(c * variables[i] for i, c in enumerate(cost) if c != 0), GRB.MINIMIZE  # pylint: disable=no-member
    )

    # Build constraints
    csr_constraints = build_coo_constraints(indices, sizes).tocsr()

    for i in range(csr_constraints.shape[0]):
        row = (
            gp.quicksum(  # pylint: disable=no-member
                variables[csr_constraints.indices[k]]
                for k in range(csr_constraints.indptr[i], csr_constraints.indptr[i + 1])
            )
            == 1
        )
        model.addConstr(row, name=f"constraint_{i}")

    # Disable logs
    model.setParam("OutputFlag", 0)
    model.optimize()

    x = np.array([variables[i].x for i in range(cost.size)])  # type: ignore

    return x > 0.5
