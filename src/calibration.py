import cv2
import numpy as np
import src.plot_tools as plot


class Calibration(object):

    def __init__(self, image):
        self.frame = image      # Calibration target images
        self.render = None      # Calibration target corners rendered on image
        self.points = None      # Sub-pixel location of points on calibration target
        self.boardDims = None   # Number of squares used in x-y for calibration routines

        # Stop criteria for calibration
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def try_approximate_corners(self, dimensions):

        found, corners = cv2.findChessboardCorners(self.frame, dimensions,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_ASYMMETRIC_GRID)
        if found:
            self.points = cv2.cornerSubPix(self.frame,
                                           corners, (11, 11), (-1, -1), self.criteria)
            self.boardDims = dimensions
        return found

    def update_points(self):
        newPoints = []
        for i in range(0, len(self.points)):
            if 20 < i < 42:
                newPoints.append(self.points[i])
        self.points = np.asarray(newPoints)
        self.boardDims = (7, 3)

    def render_points(self, img):
        tempImg = self.frame.copy()
        outImg = img.copy()
        cv2.cornerSubPix(tempImg, self.points, (11, 11), (-1, -1), self.criteria)
        cv2.drawChessboardCorners(outImg, self.boardDims, self.points, True)
        self.render = outImg


def generate_calibration(imageSet, dimms, updatePoints=False):
    for image in imageSet.images:   # Fetch an image from the image set
        data = Calibration(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Ensure that sufficient corners exist for calibration
        if data.try_approximate_corners(dimms):
            if updatePoints:                        # Update points for TASK 1.4
                data.update_points()
            data.render_points(image)               # Render points
            imageSet.calibrationSet.append(data)    # Add points to array in ImageSet
    print("Generated Calibration Data on {} Images".
          format(len(imageSet.calibrationSet)))


def calibrate_camera(imageSet):
    calibSet = imageSet.calibrationSet  # Retrive points for all images in ImageSet
    rows = calibSet[0].boardDims[0]
    cols = calibSet[0].boardDims[1]
    shape = calibSet[0].frame.shape

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Generate 2d points in image plane. Be pythonic about it
    imgPoints = [calibSet[i].points for i in range(0, len(calibSet))]

    # Allocate space in an array for 3d points in world frame
    objPoints = [objp for _ in range(0, len(calibSet))]

    # Put return values of cv2.calibrateCamera into a dictionary for later use
    ret, imageSet.calibrationParams['mtx'], \
    imageSet.calibrationParams['dist'], \
    imageSet.calibrationParams['rvecs'], \
    imageSet.calibrationParams['tvecs'] \
        = cv2.calibrateCamera(objPoints, imgPoints, shape[::-1], None, None)

    # Store 2d and 3d points on calibration target
    imageSet.calibrationParams['objPoints'] = objPoints
    imageSet.calibrationParams['imgPoints'] = imgPoints
    imageSet.calibrationParams['imageSize'] = shape
    print('Calibrated Camera')


def print_calibration_matrix(imageSet, apertureWidth, apertureHeight):

    fovx, \
    fovy, \
    focalLength, \
    principalPoint, \
    aspectRatio = cv2.calibrationMatrixValues(imageSet.calibrationParams['mtx'],
                                              imageSet.calibrationParams['imageSize'],
                                              apertureWidth,
                                              apertureHeight)

    print('FOVx:\t\t\t\t', fovx)
    print('FOVy:\t\t\t\t', fovy)
    print('Focal Length:\t\t', focalLength)
    print('Principal Point:\t', principalPoint)
    print('Aspect Ratio:\t\t', aspectRatio)
    print('Camera Matrix:')
    for c1, c2, c3 in imageSet.calibrationParams['mtx']:
        print("\t\t\t\t\t%04.2f \t|\t %04.2f \t|\t %04.2f" % (c1, c2, c3))
    print('')


def remove_distortion(imageSet, crop=True, showError=False):
    mtx = imageSet.calibrationParams['mtx']
    dist = imageSet.calibrationParams['dist']

    for image in imageSet.images:

        # Generate undistorted camera
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,
                                                          dist,
                                                          (w, h),
                                                          1,
                                                          (w, h))

        # Undistort image
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None,
                                                 newcameramtx, (w, h), 5)
        dst = cv2.remap(image.copy(), mapx, mapy, cv2.INTER_LINEAR)

        # Crop
        if crop:
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
        imageSet.undistorted.append(dst)

    if showError:
        imgPoints = imageSet.calibrationParams['imgPoints']
        objPoints = imageSet.calibrationParams['objPoints']
        rvecs = imageSet.calibrationParams['rvecs']
        tvecs = imageSet.calibrationParams['tvecs']
        tot_error = 0

        for i in range(0, len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i],
                                              rvecs[i], tvecs[i],
                                              mtx, dist)
            error = cv2.norm(imgPoints[i], imgPoints2,
                             cv2.NORM_L2) / len(imgPoints2)
            tot_error += error
        print("Total Distortion Error: ", tot_error / len(objPoints))

    print('Removed Distortion on {} Images'.format(len(imageSet.images)))


def generate_overhead(imageSet, offset, show=False):

    dims = imageSet.calibrationSet[0].boardDims

    dst = np.float32([[offset, offset],
                      [offset + 100 * (dims[0] - 1), offset],
                      [offset + 100 * (dims[0] - 1), offset + 100 * (dims[1] - 1)],
                      [offset, offset + 100 * (dims[1] - 1)]])

    # Warp the image using OpenCV warpPerspective()
    for undist in imageSet.undistorted:

        # Get undistorted shape and corners
        shape = (undist.shape[1], undist.shape[0])
        corners = cv2.findChessboardCorners(undist, dims,
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH)

        if corners[1] is not None:
            src = np.float32([corners[1][0], corners[1][dims[0] - 1],
                              corners[1][-1], corners[1][-dims[0]]])

            # Given src and dst points, calculate the perspective transform matrix
            M = cv2.getPerspectiveTransform(src, dst)
            warped = cv2.warpPerspective(undist.copy(), M, (shape[0], shape[1]))
            imageSet.rectified.append(warped)

    if show:
        for image in imageSet.rectified:
            cv2.imshow('calibrated', image)
            cv2.waitKey(0)
