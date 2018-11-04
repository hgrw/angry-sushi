import numpy as np
import math
import src.math_tools as utils
import src.calibration as cal
import src.plot_tools as plot
import cv2
import src.filter_tools as filter

class Prism(object):

    def __init__(self, top, topCent, side, sideCent):
        self.rows = top.shape[1]
        self.cols = top.shape[0]
        self.top = top
        self.topCentroid = topCent
        self.side = side
        self.sideCentoid = sideCent
        self.bottom = None
        self.bottomCentroid = None
        self.translationMatrix = None

    def generate_affine_transform(self, camPos, primitive):

        # Angle to camera from top-down perspective
        angleToCamera = math.atan2(camPos[1] + self.top.shape[1] - self.topCentroid[1],
                                   camPos[0] + self.top.shape[0] - self.topCentroid[0])

        # Translation amount for each step
        deltaX = primitive * math.sin(angleToCamera)
        deltaY = primitive * math.cos(angleToCamera)

        # Arbitrary points are chosen and then translated by deltaX and deltaY, so as to provide input required to
        # Generate affine transform matrix
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[50 + deltaX, 50 + deltaY], [200 + deltaX, 50 + deltaY], [50 + deltaX, 200 + deltaY]])

        return cv2.getAffineTransform(pts1, pts2)

    def bootstrap_base(self, camPos):

        # We are going to shift the top face two pixels at a time (1mm) towards camera centre
        primitive = 2

        # Generate affine transform to shift face
        self.translationMatrix = self.generate_affine_transform(camPos, primitive)

        # Define the number of translation steps to move. Since each step is roughly 1mm, 100 translations = 10cm
        translationSteps = 100

        # Top face to shift
        shifted = self.top.copy()

        # Cost of each translation stored in cost array. Highest value in array corresponds to ideal translation length
        cost = []

        # Translate the top 100 times and store the number of white pixels after boolean and with sides.
        for _ in range(0, translationSteps):
            shifted = cv2.warpAffine(shifted, self.translationMatrix, (self.rows, self.cols))
            cost.append(np.sum((shifted & self.side) == 255))

        # Find optimal translation distance to translate top
        distance = cost.index(max(cost))

        # Overwrite translation matrix with optimal translation distance
        self.translationMatrix = self.generate_affine_transform(camPos, 2 * distance)

        # Project top face onto bottom face :)
        self.bottom = cv2.warpAffine(self.top.copy(), self.translationMatrix, (self.rows, self.cols))

        #print('INSIDE PRISM: ', self.topCentroid)
        #cv2.imshow('top', self.top ^ self.bottom)
        #cv2.imshow('side', self.side)
        #cv2.imshow('bottom', self.bottom)
        #cv2.waitKey(0)


class Environment(object):

    def __init__(self, canvas, deadzone):

        # Add deadzones to canvas
        cv2.line(canvas, deadzone[0], deadzone[1], [0, 255, 0], 3)
        cv2.line(canvas, deadzone[2], deadzone[3], [0, 255, 0], 3)

        # Environment sensing data
        self.boardMask = None           # Board mask generated from black pixels
        self.boardMaskTd = None         # Top-down rectified board mask
        self.boardMaskFilled = None     # Board mask generated from detected corners
        self.tops = None                # Combined tops for all foam objects
        self.sides = None               # Combined sides for all foam objects
        self.shapes = []                # List of numpy arrays that contain both top and side information for each obj
        self.prisms = []                # List of prism objects, one for each foam block
        self.goals = None               # Combined goals
        self.canvas = canvas            # Canvas of environment used to plot objects
        self.goalsTd = None             # Combined goals from top-down perspective
        self.canvasTd = None            # Top-down view of canvas
        self.boardCorners = []          # Board corners

        # Camera and frame data
        self.wsOrigin = None            # Origin of workspace frame
        self.rvecs = None               # Rotation matrix between workspace and camera
        self.tvecs = None               # Translation matrix between workspace and camera
        self.mtx = None                 # Camera matrix
        self.dist = None                # Camera distortion coefficients
        self.longEdgeMm = 420           # Long edge of workspace in mm
        self.shortEdgeMm = 280          # Short edge of workspace in mm

        # Path planning and goal data
        self.start = None
        self.goal = None

    def generate_workspace(self):

        mask = self.boardMaskTd.copy()

        for prism in self.prisms:
            mask = (mask | prism.top | prism.side) & ~prism.bottom

        return mask

    def get_prisms(self):

        prisms = []

        # Generate top-side pairs for each top.
        for [top, side] in self.shapes:
            tops = filter.separate_components(top)
            sides = filter.separate_components(side)

            if len(tops) == len(sides):

                # Instantiate prism object with a top, top centroid, side and side centroid
                for i in range(0, len(tops)):
                    prisms.append(Prism(tops[i][0], tops[i][1], sides[i][0], sides[i][1]))
            else:
                print('topside error! Number of tops != number of sides, dammit')
        self.prisms = prisms

    def get_start_and_end_points(self):
        goal = filter.get_circle(self.goalsTd, 39, 42)
        if goal is not None:
            self.goal = goal
        start = filter.get_circle(self.topsTd, 35, 37)
        if start is not None:
            self.start = start

        # Render start and goal location on
        if self.start is not None:
            cv2.circle(self.canvasTd, (self.start[0], self.start[1]), self.start[2], (0, 255, 0), 2)
        if self.goal is not None:
            cv2.circle(self.canvasTd, (self.goal[0], self.goal[1]), self.goal[2], (0, 255, 0), 2)

    def update_masks(self):

        # Instantiate morphology kernel
        kernel = np.ones((6, 6), np.uint8)

        # Draw polygon between workspace corners
        mask = np.zeros((1024, 1280), dtype=np.uint8)
        pts = np.array(self.boardCorners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        self.boardMaskFilled = cv2.fillPoly(mask, [pts], 255)

        # Close masks and remove small objects
        self.tops = filter.remove_components(cv2.morphologyEx(self.tops, cv2.MORPH_CLOSE, kernel), minSize=2500)
        self.sides = filter.remove_components(cv2.morphologyEx(self.sides, cv2.MORPH_CLOSE, kernel), minSize=10000)

        # Clean up masks by removing intersections
        self.sides = (self.sides & ~self.tops) & self.boardMaskFilled
        self.tops = (self.tops & ~ self.sides) & self.boardMaskFilled
        self.shapes = [[filter.remove_components(cv2.morphologyEx(top & self.tops,
                                                                  cv2.MORPH_CLOSE, kernel), minSize=2500),
                        filter.remove_components(cv2.morphologyEx(side & self.sides, cv2.MORPH_CLOSE, kernel),
                                                 minSize=2500)] for [top, side] in self.shapes]

    def get_ws_objects(self, image, hues, rectifyMask, bThresh, wThresh, hThresh, sThresh, vThresh):
        image = cv2.bilateralFilter(image, 9, 40, 40)
        gray = ~cv2.cvtColor(filter.get_clahe(image), cv2.COLOR_BGR2GRAY) & ~rectifyMask
        hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
        self.tops = np.zeros((1024, 1280), dtype=np.uint8)
        self.sides = np.zeros((1024, 1280), dtype=np.uint8)

        for hue in hues:

            # Segment top of coloured object with hue in hues
            top = cv2.inRange(hsv,
                              (max(hue[0] - hThresh, 0),
                               max(hue[1] - sThresh, 0),
                               max(hue[2] - vThresh, 0)),
                              (min(hue[0] + hThresh, 180),
                               max(hue[1] + sThresh, 255),
                               max(hue[2] + vThresh, 255)))

            # Segment entirety of coloured object with hue in hues
            side = cv2.inRange(hsv,
                               (max(hue[0] - int(1.0 * hThresh), 0), 50, 50),
                               (min(hue[0] + int(1.0 * hThresh), 180), 255, 255))

            # Add tops and sides to cumulative masks, append individual shape data to array for each hue
            self.tops = self.tops | top
            self.sides = self.sides | side
            self.shapes.append([top, side])

        # Delete cards from shapes. We only want '3d' objects in this list
        del self.shapes[-1]

        # Segment board from image
        self.boardMask = (cv2.inRange(hsv, (0, 0, 0), (180, bThresh, bThresh)) |
                          np.asarray((gray > 220) * 255, dtype=np.uint8)) & ~rectifyMask

        # Segment goal objects from image
        self.goals = np.asarray((gray < wThresh) * 255, dtype=np.uint8) & ~rectifyMask & ~self.tops

    def get_ws_frame(self, mtx, dist):

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
            self.canvas = plot.render_origin_frame(self.canvas, numpyCorners, rvecs, tvecs, mtx, dist)
            self.rvecs = rvecs
            self.tvecs = tvecs
            self.mtx = mtx
            self.dist = dist
        except OverflowError:
            print('No line to draw')

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

        # Ensure that main loop restarts if four corners aren't found
        if len(self.boardCorners) is 4:
            return True
        else:
            return False

    def get_top_down(self):

        # Instantiate distortion kernel
        dst = np.array([
            [0, 0],
            [self.longEdgeMm * 2 - 1, 0],
            [self.longEdgeMm * 2 - 1, self.shortEdgeMm * 2 - 1],
            [0, self.shortEdgeMm * 2 - 1]], dtype="float32")

        # compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(np.array(self.boardCorners, np.float32), dst)

        # Populate environment with topdown views for all objects
        #self.boardMaskTd = cv2.warpPerspective(self.boardMask, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
        #self.sidesTd = cv2.warpPerspective(self.sides, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
        #self.topsTd = cv2.warpPerspective(self.tops, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
        self.goalsTd = cv2.warpPerspective(self.goals, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
        self.canvasTd = cv2.warpPerspective(self.canvas, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
        self.boardMaskTd= cv2.warpPerspective(self.boardMask, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))

        self.shapes = [[cv2.warpPerspective(top, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2)),
                        cv2.warpPerspective(side, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
                        ] for [top, side] in self.shapes]



        #return cv2.warpPerspective(image, M, (self.longEdgeMm * 2, self.shortEdgeMm * 2))
