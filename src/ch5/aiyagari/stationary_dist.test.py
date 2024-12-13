import unittest
from dataclasses import dataclass
import numpy as np
import stationary_dist as sd

class TestStationaryDist(unittest.TestCase):
    """ Test cases for stationary_dist.py
    """
    def test_split_prob_to_a_grid(self):
        @dataclass
        class Case:
            """ Test cases for split_prob_to_a_grid
            """
            describe: str
            avalue: float
            agrid: np.ndarray
            expected: np.ndarray
        testcase = [
            Case(
                describe="avalue is less than the minimum of agrid",
                avalue=10,
                agrid=np.array([20, 30, 40]),
                expected=np.array([1.0, 0.0, 0.0])
            ),
            Case(
                describe="avalue is greater than the maximum of agrid",
                avalue=50,
                agrid=np.array([20, 30, 40]),
                expected=np.array([0.0, 0.0, 1.0])
            ),
            Case(
                describe="avalue is between the minimum and maximum of agrid",
                avalue=25,
                agrid=np.array([20, 30, 40]),
                expected= np.array([0.5, 0.5, 0.0])
            ),
            Case(
                describe="very long agrid",
                avalue=25,
                agrid=np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]),
                expected=np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            )
        ]

        for case in testcase:
            with self.subTest(case.describe):
                self.assertTrue(np.allclose(sd.split_prob_to_a_grid(case.avalue, case.agrid), case.expected))
    
    def test_calc_weight_grid(self):
        @dataclass
        class Case:
            """ Test cases for calc_weight_grid
            """
            describe: str
            pfgrid: np.ndarray
            agrid: np.ndarray
            next_a_index: int
            expected: np.ndarray
        testcase = [
            Case(
                describe="all elements of pfgrid are the same as the minimum of agrid",
                pfgrid=np.array([[12, 12], [12, 12], [12, 12]]),
                agrid=np.array([12, 24, 36]),
                next_a_index=0, # 0th element of agrid is 12
                expected=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
            ),
            Case(
                describe="all elements of pfgrid are the same as the maximum of agrid",
                pfgrid=np.array([[36, 36], [36, 36], [36, 36]]),
                agrid=np.array([12, 24, 36]),
                next_a_index=2, # 2nd element of agrid is 36
                expected=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
            ),
            Case(
                describe="pfgrid is between the minimum and maximum of agrid",
                pfgrid=np.array([[12, 24], [24, 36], [12, 36]]),
                agrid=np.array([12, 24, 36]),
                next_a_index=1, # 1st element of agrid is 24
                expected=np.array([[0, 1], [1, 0], [0, 0]])
            ),
            Case(
                describe="pfgrid is not on the grid",
                pfgrid=np.array([[18, 18], [18, 30], [30, 30]]),
                agrid=np.array([12, 24, 36]),
                next_a_index=1, # 1st element of agrid is 24
                expected=np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
            )
        ]

        for case in testcase:
            with self.subTest(case.describe):
                self.assertTrue(np.allclose(sd.calc_weight_grid(case.pfgrid, case.agrid, case.next_a_index, sd.split_prob_to_a_grid), case.expected))
    
    def test_gen_pmesh(self):
        @dataclass
        class Case:
            """ Test cases for gen_pmesh
            """
            describe: str
            pfgrid: np.ndarray
            P: np.ndarray
            next_y_index: int
            expected: np.ndarray
        testcase = [
            Case(
                describe="stochastic exogenous variable y is binary and P is 2 times 2 matrix",
                pfgrid=np.array([[12, 12], [12, 12], [12, 12]]),
                P=np.array([[0.9, 0.1], [0.1, 0.9]]),
                next_y_index=0, # which means y_high
                expected=np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
            ),
            Case(
                describe="stochastic exogenous variable y have 3 grids and P is 3 times 3 matrix",
                pfgrid=np.array([[12, 12, 12], [12, 12, 12], [12, 12, 12]]),
                P=np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]),
                next_y_index=1, # which means y_mid
                expected=np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])
            )
        ]
        for case in testcase:
            with self.subTest(case.describe):
                self.assertTrue(np.allclose(sd.gen_pmesh(case.pfgrid, case.P, case.next_y_index), case.expected))

    def test_calc_prob_mass_func_point(self):
        @dataclass
        class Case:
            """ Test cases for calc_prob_mass_func_point
            """
            describe: str
            Pmesh: np.ndarray
            weight: np.ndarray
            sd_grid: np.ndarray
            expected: float
        testcase = [
            Case(
                describe="Simple case",
                Pmesh=np.array([[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]),
                weight=np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
                sd_grid=np.array([[0.4, 0.2], [0.1, 0.1], [0.1, 0.1]]),
                expected=0.38 # 0.9 * 1.0 * 0.4 + 0.1 * 1.0 * 0.2
            ),
        ]
        for case in testcase:
            with self.subTest(case.describe):
                self.assertAlmostEqual(sd.calc_sd_point(case.Pmesh, case.weight, case.sd_grid), case.expected)
    
    def test_calc_update_sd(self):
        @dataclass
        class Case:
            """ Test cases for update_sd
            """
            describe: str
            pfgrid: np.ndarray
            agrid: np.ndarray
            sd_grid: np.ndarray
            P: np.ndarray
            expected: float
        testcase = [
            Case(
                describe="(1)sum of calculated sd_grid is 1",
                pfgrid=np.array([[12, 12], [12, 12], [12, 12]]),
                agrid=np.array([12, 24, 36]),
                sd_grid=np.array([[0.4, 0.2], [0.1, 0.1], [0.1, 0.1]]),
                P=np.array([[0.9, 0.1], [0.1, 0.9]]),
                expected=1
            ),
            Case(
                describe="(2)sum of calculated sd_grid is 1",
                pfgrid=np.array([[12, 40], [100, 80], [30, 20]]),
                agrid=np.array([10, 30, 50]),
                sd_grid=np.array([[0.4, 0.2], [0.1, 0.1], [0.1, 0.1]]),
                P=np.array([[0.9, 0.1], [0.1, 0.9]]),
                expected=1
            ),
            
        ]
        for case in testcase:
            with self.subTest(case.describe):
                self.assertTrue(np.sum(sd.update_sd(case.pfgrid, case.agrid, case.sd_grid, case.P)), case.expected)
    
    def test_solve_stationary_dist(self):
        @dataclass
        class Case:
            """ Test cases for solve_stationaly_dist
            """
            describe: str
            pfgrid: np.ndarray
            agrid: np.ndarray
            sd_grid: np.ndarray
            P: np.ndarray
            tol: float
            max_iter: int
            expected: float
        testcase = [
            Case(
                describe="Simple case",
                pfgrid=np.array([[20, -48], [32, 15], [144, 27]]),
                agrid=np.array([12, 24, 36]),
                sd_grid=np.full((3, 2), 1/6),
                P=np.array([[0.9, 0.1], [0.1, 0.9]]),
                tol=1e-6,
                max_iter=1000,
                expected=1
            ),
        ]
        for case in testcase:
            with self.subTest(case.describe):
                self.assertTrue(np.sum(sd.solve_stationary_dist(case.pfgrid, case.agrid, case.sd_grid, case.P, case.tol, case.max_iter)), case.expected)

if __name__ == "__main__":
    unittest.main()