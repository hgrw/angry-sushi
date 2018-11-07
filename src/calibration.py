import cv2
import numpy as np
import math
import src.plot_tools as plot


class Calibration(object):

    def __init__(self, image):
        self.frame = image      # Calibration target images
        self.render = None      # Calibration target corners rendered on image
        self.corners = None     # Sub-pixel location of points on calibration target
        self.objPoints = None   # Location of corners in calibration target frame. Origin at top left corner.

        # Stop criteria for calibration
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def try_approximate_corners(self, dimensions):
        found, corners = cv2.findChessboardCorners(self.frame, dimensions,
                                                   flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                   cv2.CALIB_CB_ASYMMETRIC_GRID)
        if found:
            cv2.cornerSubPix(self.frame, corners, (11, 11), (-1, -1), self.criteria)
            self.corners = corners
        return found

    def render_points(self, img, dimms):
        outImg = img.copy()
        cv2.drawChessboardCorners(outImg, dimms, self.corners, True)
        self.render = outImg


def get_points(image, dimms):

    # Instantiate calibration object
    data = Calibration(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    # Ensure that sufficient corners exist for calibration
    if data.try_approximate_corners(dimms):
        data.render_points(image, dimms)               # Render points
        return data


def generate_origin_square(corners):

    """
    Generating the origin square will give us the rotation and translation matrices to convert image coordinates into
    world coordinates on the workspace. These matrices are saved in the .mat file to be used in kinematics routines
    by other team members.

    Workspace dimensions: 419mm X 279.5mm
    Checkerboard square dimensions: 36mm X 36mm

    fraction along long edge:   11.64 squares
    fraction along short edge:  7.76 squares

    :param corners: Python list of corners, arranged as follows: [top-left, top-right, bottom-right, bottom-left]
    :return: Numpy array of corners suitable for solvePnPRansac function. Virtual chessboard square for extrinsics!
    Note that the order of points returned is not the order given as input. This is to satisfy solvePnPRansac

    NOTE: The construction of the virtual square makes the assumption that camera perspective is an affine transform.
    **THIS IS FALSE**. However, since the square is small, the error would be a few pixels and hopefully not
    too bad.
    """
    longEdgeFraction = 11.64
    shortEdgeFraction = 7.76

    # Get board dimensions in camera frame (pixels and radians)
    topEdge = math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1])
    topAngle = math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0])
    leftEdge = math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1])
    leftAngle = math.atan2(corners[3][1] - corners[0][1], corners[3][0] - corners[0][0])

    # Get square dimensions in camera frame (pixels)
    squareEdgeTopBottom = topEdge / longEdgeFraction
    squareEdgeLeftRight = leftEdge / shortEdgeFraction

    # Project square edges into camera frame using top left board corner
    pt1 = (corners[0][0] + squareEdgeTopBottom * math.cos(topAngle),
           corners[0][1] + squareEdgeTopBottom * math.sin(topAngle))
    pt3 = (corners[0][0] + squareEdgeLeftRight * math.cos(leftAngle),
           corners[0][1] + squareEdgeLeftRight * math.sin(leftAngle))

    # Project square edge not connected to top left board corner by extending from bottom left origin-square corner
    pt2 = (pt3[0] + squareEdgeTopBottom * math.cos(topAngle), pt3[1] + squareEdgeTopBottom * math.sin(topAngle))

    return [corners[0], pt1, pt3, pt2]


def align_tilt(cam, worldCorners):
    print('ALIGN CAMERA TILT LINE WITH BOTTOM OF CAMERA MOUNT')
    while True:
        canvas = cam.get_img(rectify=True)

        # Plot markers of virual space (including region outiside camera's field of view)
        cv2.line(canvas, worldCorners[0], worldCorners[1], [0, 255, 0], 3)
        cv2.line(canvas, worldCorners[1], worldCorners[2], [0, 255, 0], 3)
        cv2.line(canvas, worldCorners[2], worldCorners[3], [0, 255, 0], 3)
        cv2.line(canvas, worldCorners[3], worldCorners[0], [0, 255, 0], 3)

        # Plot markers for positioning manipulator mount
        cv2.line(canvas, (400, 872), (400, 912), [0, 0, 255], 3)
        cv2.line(canvas, (600, 906), (600, 946), [0, 0, 255], 3)
        cv2.line(canvas, (800, 941), (800, 981), [0, 0, 255], 3)

        cv2.imshow('img', canvas)
        k = cv2.waitKey(1)
        if k == 27:    # Esc key to stop
            break

def get_nadir(img, extrinsics):

    return img


def calibrate_camera(camera, dimms):
    calibSet = camera.calibrationObjects  # Retrive points for all images in ImageSet
    rows, cols = dimms
    shape = calibSet[0].frame.shape

    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Generate 2d points in image plane. Be pythonic about it
    imgPoints = [calibSet[i].corners for i in range(0, len(calibSet))]

    # Allocate space in an array for 3d points in world frame
    objPoints = [objp for _ in range(0, len(calibSet))]

    # Put return values of cv2.calibrateCamera into a dictionary for later use
    ret, camera.calibrationParams['mtx'], \
    camera.calibrationParams['dist'], \
    camera.calibrationParams['rvecs'], \
    camera.calibrationParams['tvecs'] \
        = cv2.calibrateCamera(objPoints, imgPoints, shape[::-1], None, None)

    # Store 2d and 3d points on calibration target
    #camera.calibrationParams['objPoints'] = objPoints
    #camera.calibrationParams['imgPoints'] = imgPoints
    #camera.calibrationParams['imageSize'] = shape
    print('Calibrated Camera')


def print_calibration_matrix(camera, apertureWidth, apertureHeight):

    fovx, \
    fovy, \
    focalLength, \
    principalPoint, \
    aspectRatio = cv2.calibrationMatrixValues(camera.calibrationParams['mtx'],
                                              camera.calibrationParams['imageSize'],
                                              apertureWidth,
                                              apertureHeight)

    print('FOVx:\t\t\t\t', fovx)
    print('FOVy:\t\t\t\t', fovy)
    print('Focal Length:\t\t', focalLength)
    print('Principal Point:\t', principalPoint)
    print('Aspect Ratio:\t\t', aspectRatio)
    print('Camera Matrix:')
    for c1, c2, c3 in camera.calibrationParams['mtx']:
        print("\t\t\t\t\t%04.2f \t|\t %04.2f \t|\t %04.2f" % (c1, c2, c3))
    print('')


def remove_distortion(cameraParameters, image, crop=True, showError=False):
    mtx = np.asarray(cameraParameters['mtx'])
    dist = np.asarray(cameraParameters['dist'])

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

    if showError:
        imgPoints = cameraParameters['imgPoints']
        objPoints = cameraParameters['objPoints']
        rvecs = cameraParameters['rvecs']
        tvecs = cameraParameters['tvecs']
        tot_error = 0

        for i in range(0, len(objPoints)):
            imgPoints2, _ = cv2.projectPoints(objPoints[i],
                                              rvecs[i], tvecs[i],
                                              mtx, dist)
            error = cv2.norm(imgPoints[i], imgPoints2,
                             cv2.NORM_L2) / len(imgPoints2)
            tot_error += error
        print("Total Distortion Error: ", tot_error / len(objPoints))

    return dst


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
