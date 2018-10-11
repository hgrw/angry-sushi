import cv2
import math
import src.math_tools as utils
import numpy as np


def trim_images(input):
    new = cv2.resize(input[188:913, 187:1093, :], (1280, 1024), interpolation=cv2.INTER_CUBIC)
    return new


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

def get_edges(image, simpleMode=False):

    image = get_clahe(image)
    image = cv2.bilateralFilter(image, 9, 40, 40)
    image = cv2.medianBlur(image, 9)

    if simpleMode:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        # Remove saturated pixels
        image = color_mask(image, 50)

        # Apply threshold
        _, dst = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 255, cv2.THRESH_BINARY)

        # Close and fill image
        image = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, np.ones((6, 6), np.uint8), iterations=2)
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            cv2.drawContours(image, [cnt], 0, 255, -1)

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

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


def color_mask(img, thresh):

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
    mask = np.asarray((hsv[..., 1] < thresh) * 255, dtype=np.uint8)

    return cv2.bitwise_and(img, img, mask=mask)


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


class anisodiff2D(object):

    def __init__(self, num_iter=5, delta_t=1/7, kappa=30, option=2):

        super(anisodiff2D, self).__init__()

        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option

        self.hN = np.array([[0,1,0],[0,-1,0],[0,0,0]])
        self.hS = np.array([[0,0,0],[0,-1,0],[0,1,0]])
        self.hE = np.array([[0,0,0],[0,-1,1],[0,0,0]])
        self.hW = np.array([[0,0,0],[1,-1,0],[0,0,0]])
        self.hNE = np.array([[0,0,1],[0,-1,0],[0,0,0]])
        self.hSE = np.array([[0,0,0],[0,-1,0],[0,0,1]])
        self.hSW = np.array([[0,0,0],[0,-1,0],[1,0,0]])
        self.hNW = np.array([[1,0,0],[0,-1,0],[0,0,0]])

    def fit(self, img):

        diff_im = img.copy()

        dx=1; dy=1; dd = math.sqrt(2)

        for i in range(self.num_iter):

            nablaN = cv2.filter2D(diff_im,-1,self.hN)
            nablaS = cv2.filter2D(diff_im,-1,self.hS)
            nablaW = cv2.filter2D(diff_im,-1,self.hW)
            nablaE = cv2.filter2D(diff_im,-1,self.hE)
            nablaNE = cv2.filter2D(diff_im,-1,self.hNE)
            nablaSE = cv2.filter2D(diff_im,-1,self.hSE)
            nablaSW = cv2.filter2D(diff_im,-1,self.hSW)
            nablaNW = cv2.filter2D(diff_im,-1,self.hNW)

            cN = 0; cS = 0; cW = 0; cE = 0; cNE = 0; cSE = 0; cSW = 0; cNW = 0

            if self.option == 1:
                cN = np.exp(-(nablaN/self.kappa)**2)
                cS = np.exp(-(nablaS/self.kappa)**2)
                cW = np.exp(-(nablaW/self.kappa)**2)
                cE = np.exp(-(nablaE/self.kappa)**2)
                cNE = np.exp(-(nablaNE/self.kappa)**2)
                cSE = np.exp(-(nablaSE/self.kappa)**2)
                cSW = np.exp(-(nablaSW/self.kappa)**2)
                cNW = np.exp(-(nablaNW/self.kappa)**2)
            elif self.option == 2:
                cN = 1/(1+(nablaN/self.kappa)**2)
                cS = 1/(1+(nablaS/self.kappa)**2)
                cW = 1/(1+(nablaW/self.kappa)**2)
                cE = 1/(1+(nablaE/self.kappa)**2)
                cNE = 1/(1+(nablaNE/self.kappa)**2)
                cSE = 1/(1+(nablaSE/self.kappa)**2)
                cSW = 1/(1+(nablaSW/self.kappa)**2)
                cNW = 1/(1+(nablaNW/self.kappa)**2)

            diff_im = diff_im + self.delta_t * (

                (1/dy**2)*cN*nablaN +
                (1/dy**2)*cS*nablaS +
                (1/dx**2)*cW*nablaW +
                (1/dx**2)*cE*nablaE +

                (1/dd**2)*cNE*nablaNE +
                (1/dd**2)*cSE*nablaSE +
                (1/dd**2)*cSW*nablaSW +
                (1/dd**2)*cNW*nablaNW
            )

        return diff_im