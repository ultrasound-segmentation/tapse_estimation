# import cupy as cp
import numpy as np
import torch
from numpy.polynomial import polynomial as poly
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

""" Script that performs all the calculations of the clinical indices given the 
predicted coordinates"""


def find_parallel_direction(points):
    """
    Find the direction of the vector that is parallel to the array.
    The array should be a 2D numpy array with shape (n, 2).
    """
    if len(points) < 2:
        raise ValueError("Array must contain at least two points.")

    # Fit a linear polynomial: returns [intercept, slope]
    b, m = poly.polyfit(points[0], points[1], 1)

    # Create direction vector using the slope
    v = np.array([1, m])  # dx = 1, dy = m

    # Normalize to get the versor
    versor = v / np.linalg.norm(v)

    return versor


def find_es(window):
    """
    function that finds the es and ed frames from a window of predictions spanning from one r peak to the next one
    args:
    window: np.ndarray with shape (n, 3, 2) containing the coordinates of the 3 landmarks for each of the frames in an heartbeat

    returns:
    es: index of the frame where the fw annular point is the farthest from its ED position (frame 0).
    The index is relative to the beginning of the window.
    """
    fw = window[:, 0]
    distances_from_ed = np.linalg.norm(fw - fw[0], axis=1)
    es = distances_from_ed.argmax()

    return es


class RVCalculator:
    def __init__(self, ed_frame, es_frame, method="triangle"):
        self.ed_free_wall = ed_frame[0]
        self.ed_septum = ed_frame[1]
        self.ed_apex = ed_frame[2]

        self.es_free_wall = es_frame[0]
        self.es_septum = es_frame[1]
        self.es_apex = es_frame[2]

        self.method = method
        self._compute()

    def _compute(self):
        ed_mid = (self.ed_free_wall + self.ed_septum) / 2
        es_mid = (self.es_free_wall + self.es_septum) / 2

        self.tapse_fw = np.linalg.norm(self.ed_free_wall - self.es_free_wall)
        self.tapse_sep = np.linalg.norm(self.ed_septum - self.es_septum)

        self.ed_len_fw = np.linalg.norm(self.ed_free_wall - self.ed_apex, axis=-1)
        self.ed_len_sep = np.linalg.norm(self.ed_septum - self.ed_apex, axis=-1)
        self.ed_len_mid = np.linalg.norm(ed_mid - self.ed_apex, axis=-1)

        self.es_len_fw = np.linalg.norm(self.es_free_wall - self.es_apex, axis=-1)
        self.es_len_sep = np.linalg.norm(self.es_septum - self.es_apex, axis=-1)
        self.es_len_mid = np.linalg.norm(es_mid - self.es_apex, axis=-1)

        self.ed_diam = np.linalg.norm(self.ed_free_wall - self.ed_septum, axis=-1)
        self.es_diam = np.linalg.norm(self.es_free_wall - self.es_septum, axis=-1)

        if self.method == "triangle":
            ed_s = (self.ed_len_fw + self.ed_len_sep + self.ed_diam) / 2
            self.ed_area = np.sqrt(
                ed_s
                * (ed_s - self.ed_len_fw)
                * (ed_s - self.ed_len_sep)
                * (ed_s - self.ed_diam)
            )

            es_s = (self.es_len_fw + self.es_len_sep + self.es_diam) / 2
            self.es_area = np.sqrt(
                es_s
                * (es_s - self.es_len_fw)
                * (es_s - self.es_len_sep)
                * (es_s - self.es_diam)
            )

        elif self.method == "spline":
            self.ed_area = self._spline_area(
                self.ed_free_wall, self.ed_apex, self.ed_septum, n_points=6
            )
            self.es_area = self._spline_area(
                self.es_free_wall, self.es_apex, self.es_septum, n_points=5
            )

        else:
            raise ValueError("method must be 'triangle' or 'spline'")

        self.rvfac = (self.ed_area - self.es_area) / self.ed_area * 100

        self.strain_fw = (self.ed_len_fw - self.es_len_fw) / self.ed_len_fw * 100
        self.strain_sep = (self.ed_len_sep - self.es_len_sep) / self.ed_len_sep * 100
        self.strain_mid = (self.ed_len_mid - self.es_len_mid) / self.ed_len_mid * 100
        self.strain_global = (
            ((self.ed_len_fw + self.ed_len_sep) - (self.es_len_fw + self.es_len_sep))
            / (self.ed_len_fw + self.ed_len_sep)
            * 100
        )

    @staticmethod
    def _spline_area(p1, apex, p2, n_points):
        pts = np.vstack([p1, apex, p2, p1])
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = np.column_stack((x_new, y_new))
        return 0.5 * np.abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )


class RVCalculatorBest:
    def __init__(self, window_none, window_kalman, window_avg, window_both, ed, es):
        # extract ed and es frames for each of the types of filtering
        none_ed = window_none[ed]
        none_es = window_none[es]

        kalman_ed = window_kalman[ed]
        kalman_es = window_kalman[es]

        avg_ed = window_avg[ed]
        avg_es = window_avg[es]

        both_ed = window_both[ed]
        both_es = window_both[es]

        # none
        self.none_ed_free_wall = none_ed[0]
        self.none_ed_septum = none_ed[1]
        self.none_ed_apex = none_ed[2]

        self.none_es_free_wall = none_es[0]
        self.none_es_septum = none_es[1]
        self.none_es_apex = none_es[2]

        self.none_ed_mid = (self.none_ed_free_wall + self.none_ed_septum) / 2
        self.none_es_mid = (self.none_es_free_wall + self.none_es_septum) / 2

        # kalman
        self.kalman_ed_free_wall = kalman_ed[0]
        self.kalman_ed_septum = kalman_ed[1]
        self.kalman_ed_apex = kalman_ed[2]

        self.kalman_es_free_wall = kalman_es[0]
        self.kalman_es_septum = kalman_es[1]
        self.kalman_es_apex = kalman_es[2]

        self.kalman_ed_mid = (self.kalman_ed_free_wall + self.kalman_ed_septum) / 2
        self.kalman_es_mid = (self.kalman_es_free_wall + self.kalman_es_septum) / 2

        # avg
        self.avg_ed_free_wall = avg_ed[0]
        self.avg_ed_septum = avg_ed[1]
        self.avg_ed_apex = avg_ed[2]

        self.avg_es_free_wall = avg_es[0]
        self.avg_es_septum = avg_es[1]
        self.avg_es_apex = avg_es[2]

        self.avg_ed_mid = (self.avg_ed_free_wall + self.avg_ed_septum) / 2
        self.avg_es_mid = (self.avg_es_free_wall + self.avg_es_septum) / 2

        # both
        self.both_ed_free_wall = both_ed[0]
        self.both_ed_septum = both_ed[1]
        self.both_ed_apex = both_ed[2]

        self.both_es_free_wall = both_es[0]
        self.both_es_septum = both_es[1]
        self.both_es_apex = both_es[2]

        self.both_ed_mid = (self.both_ed_free_wall + self.both_ed_septum) / 2
        self.both_es_mid = (self.both_es_free_wall + self.both_es_septum) / 2

        self._compute()

    def _compute(self):
        self.tapse_fw = np.linalg.norm(
            self.kalman_ed_free_wall - self.kalman_es_free_wall
        )
        self.tapse_sep = np.linalg.norm(self.both_ed_septum - self.both_es_septum)

        self.ed_len_fw = np.linalg.norm(
            self.none_ed_free_wall - self.none_ed_apex, axis=-1
        )
        self.ed_len_sep = np.linalg.norm(
            self.none_ed_septum - self.none_ed_apex, axis=-1
        )
        self.ed_len_mid = np.linalg.norm(self.none_ed_mid - self.none_ed_apex, axis=-1)

        self.es_len_fw = np.linalg.norm(
            self.kalman_es_free_wall - self.kalman_es_apex, axis=-1
        )
        self.es_len_sep = np.linalg.norm(
            self.both_es_septum - self.both_es_apex, axis=-1
        )
        self.es_len_mid = np.linalg.norm(self.both_es_mid - self.both_es_apex, axis=-1)

        self.ed_diam = np.linalg.norm(
            self.kalman_ed_free_wall - self.kalman_ed_septum, axis=-1
        )
        self.es_diam = np.linalg.norm(
            self.none_es_free_wall - self.none_es_septum, axis=-1
        )

        self.ed_area = self._spline_area(
            self.both_ed_free_wall, self.both_ed_apex, self.both_ed_septum, n_points=6
        )
        self.ed_area_rvfac = self._spline_area(
            self.both_ed_free_wall, self.both_ed_apex, self.both_ed_septum, n_points=11
        )
        self.es_area = self._spline_area(
            self.kalman_es_free_wall,
            self.kalman_es_apex,
            self.kalman_es_septum,
            n_points=5,
        )
        self.es_area_rvfac = self._spline_area(
            self.kalman_es_free_wall,
            self.kalman_es_apex,
            self.kalman_es_septum,
            n_points=5,
        )

        self.rvfac = (
            (self.ed_area_rvfac - self.es_area_rvfac) / self.ed_area_rvfac * 100
        )

        # indexes to calculate rv strains
        self.kalman_ed_len_fw = np.linalg.norm(
            self.kalman_ed_free_wall - self.kalman_ed_apex, axis=-1
        )
        self.kalman_es_len_fw = np.linalg.norm(
            self.kalman_es_free_wall - self.kalman_es_apex, axis=-1
        )
        self.kalman_ed_len_sep = np.linalg.norm(
            self.kalman_ed_septum - self.kalman_ed_apex, axis=-1
        )
        self.kalman_es_len_sep = np.linalg.norm(
            self.kalman_es_septum - self.kalman_es_apex, axis=-1
        )
        self.kalman_ed_len_mid = np.linalg.norm(
            self.kalman_ed_mid - self.kalman_ed_apex, axis=-1
        )
        self.kalman_es_len_mid = np.linalg.norm(
            self.kalman_es_mid - self.kalman_es_apex, axis=-1
        )

        self.strain_fw = (
            (self.kalman_ed_len_fw - self.kalman_es_len_fw)
            / self.kalman_ed_len_fw
            * 100
        )
        self.strain_sep = (
            (self.kalman_ed_len_sep - self.kalman_es_len_sep)
            / self.kalman_ed_len_sep
            * 100
        )
        self.strain_mid = (
            (self.kalman_ed_len_mid - self.kalman_es_len_mid)
            / self.kalman_ed_len_mid
            * 100
        )
        self.strain_global = (
            (
                (self.kalman_ed_len_fw + self.kalman_ed_len_sep)
                - (self.kalman_es_len_fw + self.kalman_es_len_sep)
            )
            / (self.kalman_ed_len_fw + self.kalman_ed_len_sep)
            * 100
        )

    @staticmethod
    def _spline_area(p1, apex, p2, n_points):
        pts = np.vstack([p1, apex, p2, p1])
        tck, u = interpolate.splprep([pts[:, 0], pts[:, 1]], s=0, per=True, k=3)
        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = interpolate.splev(u_new, tck)
        poly = np.column_stack((x_new, y_new))
        return 0.5 * np.abs(
            np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
            - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
        )
