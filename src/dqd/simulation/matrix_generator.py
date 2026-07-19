"""
CapacitanceMatrixGenerator — generates random capacitance matrices
from interval specifications.  Absorbs the old matrix_utils.py.
"""
import random
from typing import Dict, List, Optional


class CapacitanceMatrixGenerator:
    """
    Generates random capacitance matrices (symmetric and general) from
    interval specifications, with optional seed-based reproducibility.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_symmetric(
        self,
        intervals: Dict[str, List[float]],
        size: int,
        labels: List[List[str]],
    ) -> List[List[float]]:
        """
        Generate a symmetric square matrix whose entries are drawn uniformly
        from the given intervals.

        Parameters
        ----------
        intervals : dict  {label: [min, max]}
        size      : int   side length of the square matrix
        labels    : list  size×size label grid (upper triangle mirrors lower)
        """
        rng = random.Random(self.seed)
        matrix = [[0.0] * size for _ in range(size)]
        for i in range(size):
            for j in range(i, size):
                label = labels[i][j]
                lo, hi = intervals.get(label, [0, 1])
                value = round(rng.uniform(lo, hi), 4)
                matrix[i][j] = matrix[j][i] = value
        return matrix

    def generate_general(
        self,
        intervals: Dict[str, List[float]],
        shape: List[int],
        labels: List[List[str]],
    ) -> List[List[float]]:
        """
        Generate a general (non-symmetric) matrix.

        Parameters
        ----------
        intervals : dict       {label: [min, max]}
        shape     : [rows, cols]
        labels    : list       rows×cols label grid
        """
        rng = random.Random(self.seed)
        rows, cols = shape
        matrix = [[0.0] * cols for _ in range(rows)]
        for i in range(rows):
            for j in range(cols):
                label = labels[i][j]
                lo, hi = intervals.get(label, [0, 1])
                matrix[i][j] = round(rng.uniform(lo, hi), 4)
        return matrix

    # ------------------------------------------------------------------
    # Convenience: generate all four DQD matrices at once
    # ------------------------------------------------------------------

    def generate_all(self, config) -> Dict[str, List]:
        """
        Generate Cdd, Cgd, Cds, Cgs from a CapacitanceConfig instance.

        Returns a dict with keys "Cdd", "Cgd", "Cds", "Cgs".
        """
        intervals = config.intervals
        return {
            "Cdd": self.generate_symmetric(
                intervals["Cdd"], size=2, labels=config.labels_Cdd
            ),
            "Cgd": self.generate_general(
                intervals["Cgd"], shape=[2, 3], labels=config.labels_Cgd
            ),
            "Cds": self.generate_general(
                intervals["Cds"], shape=[1, 2], labels=config.labels_Cds
            ),
            "Cgs": self.generate_general(
                intervals["Cgs"], shape=[1, 3], labels=config.labels_Cgs
            ),
        }

    # ------------------------------------------------------------------
    # Debug helper
    # ------------------------------------------------------------------

    @staticmethod
    def display(matrix: List[List[float]], labels: List[List[str]], name: str) -> None:
        """Pretty-print a matrix with its labels."""
        print(f"\nGenerated {name} Matrix:")
        for i, row in enumerate(matrix):
            row_strs = [f"{labels[i][j]}={matrix[i][j]}" for j in range(len(row))]
            print("[ " + ", ".join(row_strs) + " ]")

# =====================================================================
# HOW IT WORKS:
# 1. RANDOM SAMPLING VIA SEED: The generator class can accept a 'seed' 
#    integer. This guarantees that the "random" physics values generated 
#    can be identically reproduced on any machine during testing.
#
# 2. TWO MATRIX STRATEGIES:
#      - Symmetric Generation: Used for things like dot-to-dot interactions 
#        (Cdd) where the effect of Dot 1 on Dot 2 is identical to Dot 2 on 
#        Dot 1. It only generates the upper half and mirrors it to the lower half.
#      - General Generation: Used for standard grids (like gates-to-dots) 
#        where dimensions are non-square and values are fully independent.
#
# 3. RANGE READING: It loops through the row/column layout, looks up the 
#    text label (e.g., 'd1g1') in the configuration file, finds its allowed 
#    [min, max] range, and picks a uniform random float rounded to 4 decimals.
#
# 4. MASTER ENGINE ('generate_all'): This maps directly to the companion config 
#    file. It passes the specific shapes (e.g., 2x2 for Cdd, 2x3 for Cgd) 
#    and processes all 4 required physics matrices in a single execution block.
# =====================================================================