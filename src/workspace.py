import numpy as np
import cv2
from scipy.spatial import distance as dist
import src.filter_tools as filter

class Environment(object):
    def __init__(self):
        self.boardMask = None
        self.boardCorners = None
        self.tops = None
        self.sides = None

    def get_board_corners(self):
        h, w = self.boardMask.shape
        t, b = np.array(np.where(self.boardMask == 255))[:, [0, -1]].T
        l, r = np.array(np.where(np.rot90(self.boardMask) == 255))[:, [0, -1]].T

        self.boardCorners = [tuple(reversed(t)), tuple(reversed(b)),
                             tuple([w - l[0], l[1]]), tuple([w - r[0], r[1]])]

    def four_point_transform(self, image):

        corners = [self.boardCorners[0], self.boardCorners[2], self.boardCorners[1], self.boardCorners[3]]

        width = 594     # 2 X 297mm (a4 page width)
        height = 420    # 2 X 210mm (a4 page height)

        dst = np.array([
            [0, 0],
            [594 - 1, 0],
            [594 - 1, 420 - 1],
            [0, 420 - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(np.array(corners, np.float32), dst)
        warped = cv2.warpPerspective(image, M, (width, height))

        # return the warped image
        return warped
