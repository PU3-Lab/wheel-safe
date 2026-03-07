import numpy as np

from v3.road_mask_pipeline import RoadMaskPipeline
from v3.slope_estimator import SlopeEstimator

COLOR_MAP = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]


class SlopePipeline:
    def __init__(self, config_path):
        self.session = RoadMaskPipeline()
        self.slop_estimator = SlopeEstimator(config_path)

    def run(self, img_path):
        self.pred, org_size = self.session.run(img_path)

        sv_img = np.zeros((org_size[0], org_size[1], 3), dtype=np.uint8)
        for i, color in enumerate(COLOR_MAP):
            sv_img[self.pred == i] = color

        return self.pred, sv_img

    def estimate(self, conf_map, disp_map):
        return self.slop_estimator.run(self.pred, conf_map, disp_map)
