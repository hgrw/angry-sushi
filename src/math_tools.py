import numpy as np
import math
import imutils
import cv2
from collections import defaultdict
from scipy.spatial import distance as dist


def rotate_image(image):
    h, w = image.shape

    # Double image dimensions and rotate by 45 degrees to reliably find corners
    rotated = image.copy()
    rotated = np.c_[rotated, np.zeros((h, int(w/2)), dtype=np.uint8)]
    rotated = np.c_[np.zeros((h, int(w/2)), dtype=np.uint8), rotated]
    rotated = np.r_[rotated, np.zeros((int(h/2), w * 2), dtype=np.uint8)]
    rotated = np.r_[np.zeros((int(h/2), w * 2), dtype=np.uint8), rotated]
    return imutils.rotate(rotated, 45)


def unrotate_corners(corners, image):

    vectors = [(corner) for corner in corners]
    unrotated = imutils.rotate(image, -45)
    cv2.imshow('rotated', image)
    cv2.imshow('unrotated', unrotated)
    print(corners)

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")

def draw_bounding_boxes(image, features):

    out = image.copy()

    if features is not None:
        for feat in features:
            cv2.line(out, tuple(feat[0][0]), tuple(feat[0][1]), (0, 255, 0), 5)
            cv2.line(out, tuple(feat[0][1]), tuple(feat[0][2]), (0, 255, 0), 5)
            cv2.line(out, tuple(feat[0][2]), tuple(feat[0][3]), (0, 255, 0), 5)
            cv2.line(out, tuple(feat[0][3]), tuple(feat[0][0]), (0, 255, 0), 5)
            centroid = (int(feat[1][0][0]), int(feat[1][0][1]))
            x2 = centroid[0] + 300 * math.cos(feat[1][2])
            y2 = centroid[1] + 300 * math.sin(feat[1][2])
            cv2.line(out, centroid, (int(x2), int(y2)), (0, 0, 255), 7)
            #cv2.putText(out, str(centroid),
            #            centroid,
            #            cv2.FONT_HERSHEY_DUPLEX, 4, (0, 255, 0))


    return out


def get_statistics(corners):

    #Points are given in order from top left, counter clockwise around box
    sum_x = corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]
    sum_y = corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]

    # Calculate centroid
    centroid = (sum_x / 4, sum_y / 4)

    # Calculate orientation of long edge
    orientation = math.atan2(corners[2][1] - corners[3][1],
                             corners[2][0] - corners[3][0])
    print(centroid)

    return centroid, \
           abs(corners[0][0] - corners[1][0]) * abs(corners[1][1] - corners[2][1]), \
           orientation


def get_corners(points):

    corners = []

    # We get our first corner for free
    corners.append(points[0])

    dist = math.pi / 5
    while len(corners) < 4:
        for point in points:
            uniqueAngle = True
            for corner in corners:
                if abs(corner[2] - point[2]) < dist:
                    uniqueAngle = False
            if uniqueAngle:
                corners.append(point)
                if len(corners) == 4:
                    return corners
                break
        dist -= 0.01


def distance(p1, p2):
    return math.sqrt(abs(p1[0] - p2[0]) ** 2 + abs(p1[1] - p2[1]) ** 2)


def distance_less_than(pt, extLeft, extRight, extTop, extBot, dist):

    if math.sqrt(abs((pt[0] - extLeft[0]) ** 2 - (pt[1] - extLeft[1]) ** 2)) < dist or \
            math.sqrt(abs((pt[0] - extRight[0]) ** 2 - (pt[1] - extRight[1]) ** 2)) < dist or \
            math.sqrt(abs((pt[0] - extTop[0]) ** 2 - (pt[1] - extTop[1]) ** 2)) < dist or \
            math.sqrt(abs((pt[0] - extBot[0]) ** 2 - (pt[1] - extBot[1]) ** 2)) < dist:
        return True
    else:
        return False


def get_rect(img, line=False):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    #print(img.shape)
    #print('img[', cmin, ', ', rmin, ']', img[cmin, rmin], img[rmin, cmin],
    #      img[cmax, rmax], img[rmax, cmax], img[cmin, cmax], img[rmin, rmax])
    #print(cmin, rmin, cmax, rmax)

    if line and not img[rmin, cmin]:
        return rmax, rmin, cmin, cmax

    return rmin, rmax, cmin, cmax

def segment_by_angle_kmeans(lines, k=2, **kwargs):
    """Groups lines based on angle with k-means.

    Uses k-means on the coordinates of the angle on the unit circle
    to segment `k` angles inside `lines`.
    """

    # Define criteria = (type, max_iter, epsilon)
    default_criteria_type = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
    criteria = kwargs.get('criteria', (default_criteria_type, 10, 1.0))
    flags = kwargs.get('flags', cv2.KMEANS_RANDOM_CENTERS)
    attempts = kwargs.get('attempts', 10)

    # returns angles in [0, pi] in radians
    angles = np.array([line[0][1] for line in lines])

    # multiply the angles by two and find coordinates of that angle
    pts = np.array([[np.cos(2*angle), np.sin(2*angle)]
                    for angle in angles], dtype=np.float32)

    # run kmeans on the coords
    labels, centers = cv2.kmeans(pts, k, None, criteria, attempts, flags)[1:]
    labels = labels.reshape(-1)  # transpose to row vec

    # segment lines based on their kmeans label
    segmented = defaultdict(list)
    for i, line in zip(range(len(lines)), lines):
        segmented[labels[i]].append(line)
    segmented = list(segmented.values())
    return segmented


def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [[x0, y0]]


def segmented_intersections(lines):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i, group in enumerate(lines[:-1]):
        for next_group in lines[i+1:]:
            for line1 in group:
                for line2 in next_group:
                    intersections.append(intersection(line1, line2))

    return intersections