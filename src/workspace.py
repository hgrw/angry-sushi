import numpy as np
import imutils
import math
import src.math_tools as utils
import src.calibration as cal
import src.plot_tools as plot
import cv2
from scipy.spatial import distance as dist
import src.filter_tools as filter

class Environment(object):
    def __init__(self):
        self.boardMask = None
        self.boardMaskFilled = None
        self.tops = None
        self.sides = None
        self.goals = None
        self.boardCorners = []
        self.wsOrigin = None
        self.rvecs = None
        self.tvecs = None
        self.mtx = None
        self.dist = None
        self.longEdgeMm = 420
        self.shortEdgeMm = 280

    def update_masks(self):

        # Draw polygon between workspace corners
        mask = np.zeros((1024, 1280), dtype=np.uint8)
        pts = np.array(self.boardCorners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.boardMaskFilled = cv2.fillPoly(mask, [pts], 255)

        # Close masks and remove small objects
        self.tops = filter.remove_components(cv2.morphologyEx(self.tops, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)),
                                             minSize=2500)
        self.sides = filter.remove_components(cv2.morphologyEx(self.sides, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)),
                                              minSize=10000)

        # Clean up masks by removing intersections
        self.sides = (self.sides & ~self.tops) & self.boardMaskFilled
        self.tops = (self.tops & ~ self.sides) & self.boardMaskFilled

    def get_workspace_objects(self, image, hues, rectifyMask, bThresh, wThresh, hThresh, sThresh, vThresh):
        image = cv2.bilateralFilter(image, 9, 40, 40)
        gray = ~cv2.cvtColor(filter.get_clahe(image), cv2.COLOR_BGR2GRAY) & ~rectifyMask
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        self.tops = np.zeros((1024, 1280), dtype=np.uint8)
        self.sides = np.zeros((1024, 1280), dtype=np.uint8)

        for hue in hues:
            # Add object tops for each hue to combined tops mask
            self.tops = self.tops | cv2.inRange(hsv,
                                                (max(hue[0] - hThresh, 0),
                                                 max(hue[1] - sThresh, 0),
                                                 max(hue[2] - vThresh, 0)),
                                                (min(hue[0] + hThresh, 180),
                                                 max(hue[1] + sThresh, 255),
                                                 max(hue[2] + vThresh, 255)))

            # Add entirety of coloured prisms and cards
            self.sides = self.sides | cv2.inRange(hsv,
                                                  (max(hue[0] - int(1.0 * hThresh), 0), 50, 50),
                                                  (min(hue[0] + int(1.0 * hThresh), 180), 255, 255))

        # Segment board from image
        self.boardMask = (cv2.inRange(hsv, (0, 0, 0), (180, bThresh, bThresh)) |
                          np.asarray((gray > 220) * 255, dtype=np.uint8)) & ~rectifyMask

        # Segment goal objects from image
        self.goals = np.asarray((gray < wThresh) * 255, dtype=np.uint8) & ~rectifyMask & ~self.tops

    def get_workspace_frame(self, img, mtx, dist):

        # Workspace corners in workspace frame [0, 1]
        objp = np.zeros((4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)

        # SolvePnPRansac wants a very specific matrix dimension for corners. Manipulate corner matrix to conform.
        # Specifically, it is expecting a square with same dimensions as checkerboard square. Since we know board
        # dimensions, we can generate such a square on which our origin lays. Then it must be manipulated so that
        # it conforms to the pedantic cv2 specification. To be fair, it because cv2 is so well optimised.
        self.wsOrigin = cal.generate_origin_square(self.boardCorners)
        numpyCorners = np.expand_dims(np.asarray([np.asarray(cnr, dtype=np.float32).T for cnr in self.wsOrigin]),
                                      axis=1)

        # Generate rotation and translation vectors for calibration target
        _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, numpyCorners, mtx, dist)

        try:
            img = plot.render_origin_frame(img, numpyCorners, rvecs, tvecs, mtx, dist)
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.mtx = mtx
            self.dist = dist
        except OverflowError:
            print('No line to draw')
        return img

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

        rotated = utils.rotate_image(self.boardMask)
        h, w = rotated.shape

        try:
            t, b = np.array(np.where(rotated == 255))[:, [0, -1]].T
            l, r = np.array(np.where(np.rot90(rotated) == 255))[:, [0, -1]].T

            rotatedCorners = [tuple(reversed(t)), tuple(reversed(b)),
                              tuple([w - l[0], l[1]]), tuple([w - r[0], r[1]])]

            unrotatedVectors = [(math.hypot(1279.0 - cnr[0], cnr[1] - 1023.0),
                               math.atan2(1023.0 - cnr[1], 1279.0 - cnr[0]) - math.pi / 4) for cnr in rotatedCorners]
            self.boardCorners = [(640 + int(vect[0] * math.sin(vect[1])),
                                  512 - int(vect[0] * math.cos(vect[1]))) for vect in unrotatedVectors]

            self.boardCorners = [self.boardCorners[3], self.boardCorners[0], self.boardCorners[2], self.boardCorners[1]]


        except IndexError:
           print('Not enough board pixels to get corners')

    def get_top_down(self, image):

        # Instantiate distortion kernel
        dst = np.array([
            [0, 0],
            [self.longEdgeMm * 2 - 1, 0],
            [self.longEdgeMm * 2 - 1, self.shortEdgeMm * 2 - 1],
            [0, self.shortEdgeMm * 2 - 1]], dtype="float32")

        # compute the perspective transform matrix and then apply it
        M = cv2.getPerspectiveTransform(np.array(self.boardCorners, np.float32), dst)

        return cv2.warpPerspective(image, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
