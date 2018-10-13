import numpy as np
import cv2
from scipy.spatial import distance as dist
import src.filter_tools as filter

class Environment(object):
    def __init__(self):
        self.worldMask = None
        self.boardMask = None
        self.shapeMask = None
        self.boardCorners = None

    def get_board_corners(self):
        _, contours, _ = cv2.findContours(self.boardMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours is not None:
            cnt = contours[0]
            self.boardCorners = [tuple(cnt[cnt[:, :, 0].argmin()][0]), tuple(cnt[cnt[:, :, 0].argmax()][0]),
                                 tuple(cnt[cnt[:, :, 1].argmin()][0]), tuple(cnt[cnt[:, :, 1].argmax()][0])]

