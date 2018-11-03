import numpy as np
import imutils
import math
import src.math_tools as utils
import cv2
from scipy.spatial import distance as dist
import src.filter_tools as filter

class Environment(object):
    def __init__(self):
        self.boardMask = None
        self.boardFilled = None
        self.boardCorners = None
        self.tops = None
        self.sides = None

    def fill_board(self):

        # Prepare filled image of board (i.e. fill shape gaps). Don't overwrite board.
        self.boardFilled = self.boardMask.copy()

        # Mask used to flood filling. Notice the size needs to be 2 pixels than the image.
        h, w = self.boardMask.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(self.boardFilled, mask, (0, 0), 255)

        # Combine the two images to get the foreground.
        self.boardFilled = self.boardMask | cv2.bitwise_not(self.boardFilled)

    def get_board_corners(self):

        # Remove shape holes from board surface
        #self.fill_board()

        # Morphology kernel
        #kernel = np.ones((5, 5), np.uint8)

        # Canvas to store corners
        #corners = np.zeros(self.boardMask.shape, np.uint8)

        # Find Harris corners
        #dst = cv2.cornerHarris(gray, 4, 1, 0.01)

        # result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst, None)

        # Threshold for an optimal value, it may vary depending on the image.
        #corners[dst > 0.01 * dst.max()] = 255

        # Remove corners not on edge of board
        #corners = cv2.dilate(corners & cv2.dilate(cv2.morphologyEx(self.boardFilled,
        #                                                           cv2.MORPH_GRADIENT,
        #                                                           kernel,
        #                                                           iterations=2), kernel), kernel)

        #numComponents, output, stats, centroids = cv2.connectedComponentsWithStats(corners, connectivity=8)


        #cv2.imshow('board', self.boardFilled)
        #cv2.imshow('corners', corners)
        rotated = utils.rotate_image(self.boardMask)
        h, w = rotated.shape
        print((h, w), self.boardMask.shape)
        cv2.imshow('original', self.boardMask)

        try:
            t, b = np.array(np.where(rotated == 255))[:, [0, -1]].T
            l, r = np.array(np.where(np.rot90(rotated) == 255))[:, [0, -1]].T

            rotatedCorners = [tuple(reversed(t)), tuple(reversed(b)),
                              tuple([w - l[0], l[1]]), tuple([w - r[0], r[1]])]

            unrotatedVectors = [(math.hypot(1279.0 - cnr[0], cnr[1] - 1023.0),
                               math.atan2(1023.0 - cnr[1], 1279.0 - cnr[0]) - math.pi / 4) for cnr in rotatedCorners]
            self.boardCorners = [(640 + int(vect[0] * math.sin(vect[1])),
                                  512 - int(vect[0] * math.cos(vect[1]))) for vect in unrotatedVectors]

        except IndexError:
           print('Not enough board pixels to get corners')

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
