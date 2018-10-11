import cv2
import numpy as np
import math


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
