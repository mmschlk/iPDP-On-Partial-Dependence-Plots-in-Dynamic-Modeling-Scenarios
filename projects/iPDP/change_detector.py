"""This module contains the PDPChangeDetector class."""

import numpy as np
from river.drift import ADWIN


class PDPChangeDetector:

    def __init__(self, grid_size: int, change_detector: ADWIN = None):
        self.grid_size = grid_size
        if change_detector is None:
            change_detector = ADWIN(delta=0.00001, grace_period=2000)
        self.change_detector: ADWIN = change_detector
        self.seen_samples: int = 0
        self.history_x = []
        self.history_y = []

    def _update_detectors(self, pdp_x, pdp_y):
        grid_points_x, grid_points_y = pdp_x.values(), pdp_y.values()
        mean_centered_pd = np.mean(list(grid_points_y))
        pdp_fi = 0
        for n, (grid_point_x, grid_point_y) in enumerate(zip(grid_points_x, grid_points_y)):
            mean_point_pd = grid_point_y - mean_centered_pd
            pdp_fi += mean_point_pd ** 2
        self.change_detector.update(pdp_fi)
        self.history_x.append(pdp_x)
        self.history_y.append(pdp_y)

    def detect_one(self, pdp_x, pdp_y):
        drift_detected = False
        if len(pdp_x) > 0 and len(pdp_y) > 0:
            self._update_detectors(pdp_x, pdp_y)
            if self.change_detector.drift_detected:
                drift_detected = True
            self.seen_samples += 1
        return drift_detected