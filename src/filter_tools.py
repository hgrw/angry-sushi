import cv2
import math
import src.math_tools as utils
import src.plot_tools as plot
import numpy as np


def get_min_rects(edge, image):

    numComponents, output, stats, centroids = cv2.connectedComponentsWithStats(edge, connectivity=8)

    # for every component in the image, you keep it only if it's above min_size
    for i in range(1, numComponents):
        contours, hierarchy = cv2.findContours(edge, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        print(M)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        cv2.imshow(image)
        cv2.waitKey(0)

def get_boxes(edge):

    features = []   # [centroid, corners]
    lines = cv2.HoughLines(edge, 1, np.pi / 180, 40)

    if lines is not None:
        segmented = utils.segment_by_angle_kmeans(lines)
        intersections = utils.segmented_intersections(segmented)

        # Analyse each connected object separately
        numComponents, output, stats, centroids = cv2.connectedComponentsWithStats(edge, connectivity=8)

        # for every component in the image, you keep it only if it's above min_size
        for i in range(1, numComponents):
            points = []
            comp = np.zeros(edge.shape)
            comp[output == i] = 255
            comp = cv2.dilate(comp, np.ones((5, 5), np.uint8), iterations=3)

            # Refine intersections so that they occur only on edges
            for pt in intersections:
                if comp[min(max(pt[0][1], 0), edge.shape[0] - 1), min(max(pt[0][0], 0), edge.shape[1] - 1)] == 255:
                    angle = (math.atan2(pt[0][1] - centroids[i][1], pt[0][0] - centroids[i][0]) + math.pi) % (2 * math.pi)
                    points.append([tuple(pt[0]),
                                   utils.distance(pt[0], centroids[i]),
                                   angle])
            if len(points) > 3:
                corners = utils.get_corners(sorted(points, key=lambda l: l[1], reverse=True))
                corners = utils.order_points(np.asarray([corners[0][0], corners[1][0], corners[2][0], corners[3][0]]))
            else:
                corners = None
            if corners is not None:
                statistics = utils.get_statistics(corners)
                if statistics[1] > 20000:
                    features.append([corners, statistics])
        return features
    else:
        return None


def get_elements(image, hues, rectifyMask, bThresh, hThresh, sThresh, vThresh):
    image = cv2.bilateralFilter(image, 9, 40, 40)
    hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)
    boardMask = np.zeros(image[:, :, 0].shape, dtype=np.uint8)
    topMasks = np.zeros(image[:, :, 0].shape, dtype=np.uint8)
    sideMasks = np.zeros(image[:, :, 0].shape, dtype=np.uint8)

    tops = []
    shapes = []
    for hue in hues:
        topMask = cv2.inRange(hsv,
                              (max(hue[0] - hThresh, 0),
                               max(hue[1] - sThresh, 0),
                               max(hue[2] - vThresh, 0)),
                              (min(hue[0] + hThresh, 180),
                               max(hue[1] + sThresh, 255),
                               max(hue[2] + vThresh, 255)))
        shapeMask = cv2.inRange(hsv,
                                (max(hue[0] - int(1.0 * hThresh), 0), 50, 50),
                                (min(hue[0] + int(1.0 * hThresh), 180), 255, 255))
        boardMask = boardMask | cv2.inRange(hsv, (0, 0, 0), (180, bThresh, bThresh)) & ~rectifyMask
        topMasks = topMasks | topMask
        sideMasks = (sideMasks | shapeMask) & ~topMasks & ~boardMask

        tops.append(cv2.morphologyEx(remove_components(topMask, minSize=10000),
                                     cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)))
        shapes.append(cv2.morphologyEx(remove_components(shapeMask, minSize=10000),
                                       cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8)))

    # Remove lines on each image edge, caused by image rectification
    boardMask[0, :] = 0
    boardMask[-1, :] = 0
    boardMask[:, 0] = 0
    boardMask[:, -1] = 0
    return cv2.morphologyEx(boardMask, cv2.MORPH_CLOSE, np.ones((8, 8), np.uint8)), sideMasks, topMasks, tops, shapes


def get_circle(mask, minRad, maxRad):
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=8, minRadius=minRad, maxRadius=maxRad)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            if mask[i[1], i[0]] == 255:
                return i


def get_box(edges, image):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 180)

    if lines is not None:
        if len(lines) == 4:
            for line in lines:
                for rho, theta in line:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 2000 * (-b))
                    y1 = int(y0 + 2000 * (a))
                    x2 = int(x0 - 2000 * (-b))
                    y2 = int(y0 - 2000 * (a))

                    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        #cv2.imshow('lines', image)


def get_edges(image):

    (mu, sigma) = cv2.meanStdDev(image)
    return cv2.Canny(cv2.medianBlur(image, 9), mu - sigma, mu + sigma, 20, L2gradient=True)


def infill_components(image):
    #cv2.imshow('image', image)
    #cv2.waitKey(0)
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(image, mask, (0, 0), 255)
    return ~image


def color_mask(img, thresh, binary=False):

    """
    Remove pixels that have saturation greater than threshold
    :param img: input image
    :param thresh: [uint8] threshold value
    :return:
    """

    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        print('hsv image error')
        return img

    mask = np.asarray((hsv[..., 1] > thresh) * 255, dtype=np.uint8)

    if binary:
        return mask

    return cv2.bitwise_and(img, img, mask=mask)

def separate_components(mask):

    # Output list to populate
    components = []

    # find all connected components
    num, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Store connected components separately
    for i in range(1, num):

        out = np.zeros(mask.shape, dtype=np.uint8)
        out[output == i] = 255
        components.append([out, centroids[i]])

    return components

def remove_components(image, largest=None, minSize=None, minWidth=None):

    # find all your connected components
    numComponents, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # your answer image
    out = np.zeros(image.shape)

    largestArea = 0
    largestIndex = 0
    # for every component in the image, you keep it only if it's above min_size
    for i in range(1, numComponents):

        if largest:
            if stats[i, cv2.CC_STAT_AREA] >= largestArea:
                largestArea = stats[i, cv2.CC_STAT_AREA]
                largestIndex = i

        if minSize is not None:
            if stats[i, cv2.CC_STAT_AREA] >= minSize:
                out[output == i] = 255

        if minWidth is not None:
            if stats[i, cv2.CC_STAT_WIDTH] * stats[i, cv2.CC_STAT_HEIGHT] >= minWidth:
                out[output == i] = 255

    if largest:
        out[output == largestIndex] = 255

    return np.asarray(out, dtype=np.uint8)

def get_clahe(input, tileGridSize=(8, 8), clipLimit=3.0):
    """
    Perform Contrast Limited Adaptive Histogram Equalisation.

    :param input: [uint8], BGR image
    :param tileSize: tuple, to set kernel dimensions
    :return: [uint8], BGR image
    """

    lab = cv2.cvtColor(input, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgrCorr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return bgrCorr

