import cv2
import numpy as np
import math


def render_origin_frame(img, corners, rvecs, tvecs, mtx, dist):

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    # project 3D points to image plane
    imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def show_mask(img, mask, chan):
    bgr = img.copy()
    bgr[:, :, :] = 0
    bgr[:, :, chan] = mask
    return cv2.addWeighted(img, 1.0, bgr, 0.4, 0)


def view_set(images, resolution):

    # minimum height and width to display all images
    stacks = math.ceil(math.sqrt(len(images)))

    # append blank images to input such that length becomes square
    for _ in range(len(images), stacks ** 2):
        images.append(np.zeros(images[0].shape, dtype=np.uint8))

    # Distribute images in evenly spaced rows and columns
    rows = []
    for row in range(0, stacks):
        rows.append(np.concatenate([images[i] for i in range(row * stacks, (row + 1) * stacks)], axis=1))

    return cv2.resize(np.vstack(rows), resolution)


def plot_path(canvas, path):
    (h, w, _) = canvas.shape
    if len(path) > 0:
        for point in range(1, len(path)):
            cv2.line(canvas,
                     (int(path[point - 1][0] * w), int(path[point - 1][1] * h)),
                     (int(path[point][0] * w), int(path[point][1] * h)), [0, 255, 0], 2)
    return canvas


def view_pair(image1, image2, downSamp=True):

    # Assert image1 is colour
    try:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    except:
        pass

    # Assert image2 is colour
    try:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    except:
        pass

    if downSamp:
        return cv2.resize(np.concatenate((image1, image2), axis=1),
                          (image1.shape[1], int(image1.shape[0] / 2)))
    else:
        return np.concatenate((image1, image2), axis=1)
