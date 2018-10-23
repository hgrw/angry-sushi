import src.calibration as calibrate
import src.filter_tools as filter
import src.plot_tools as plot
import src.math_tools as utils
import src.tree as tree
import cv2
import numpy as np
import argparse
import json
import os

from src.camera import Camera
from src.workspace import Environment


def main():

    # Calibration parameters
    targetDimensions = (6, 9)
    #exposure = 10000
    exposure = 40000
    hThresh = 10
    sThresh = 10
    vThresh = 10
    #bThresh = 70
    bThresh = 90
    testStart = (355, 355)
    testEnd = (677, 677)
    pathStep = 0.075
    pathing = False

    # Get image from camera
    cam = Camera(targetDimensions, exposure)
    env = Environment()
    cam.hardware_white_balance()

    # Load camera parameters
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData.json')
    print("LOADING CAMERA PARAMETERS")
    with open(jsonFile, 'r') as fp:
        cam.calibrationParams = json.load(fp)

    #cam.calibrate_lens()
    #cam.record_video('/home/mars/Videos/nightrider_local.avi')

    while True:
        img = cam.get_img(rectify=False)

        # Extract elements by colour
        env.boardMask, env.sides, env.tops = filter.get_elements(img.copy(),
                                                                 cam.get_object_hues(),
                                                                 bThresh, hThresh, sThresh, vThresh)

        # Remove extraneous components from edge of image
        env.boardMask = env.boardMask & filter.remove_components(
            cv2.dilate(env.boardMask, np.ones((15, 15), np.uint8), iterations=3), largest=True)

        env.get_board_corners()
        for point in env.boardCorners:
            cv2.circle(img, point, 15, [0, 255, 0])

        cv2.circle(img, testStart, 10, [255, 0, 0])
        cv2.circle(img, testEnd, 10, [255, 0, 0])

        canvas = plot.show_mask(img, env.boardMask, 2)

        if pathing:
            canvas = env.four_point_transform(canvas)
            path = tree.generate_path(env.boardMask, testStart, testEnd, pathStep)
            if path is not None:
                canvas = plot.plot_path(canvas, path)
        # Create a
        #env.shapeMask = filter.get_shapes(img, env.worldMask ^ env.boardMask)
        #board = shapes ^ ~env.boardMask

        #edges = filter.infill_components(workSpace)

        #img = filter.get_edges(img, saturationThreshold)
        #img = filter.get_clahe(img)
        #saturationMask = filter.color_mask(img, saturationThreshold)

        #img = filter.remove_components()
        #cv2.imshow("Contours", plot.view_pair(env.boardMask, env.shapeMask))
        cv2.imshow("Contours", canvas)

        k = cv2.waitKey(1)
        if k == 115:    # Esc key to stop
            print("PATHING MODE ENGAGED")
            pathing = True
    exit(0)

    # Generate top-down view of image set
    cal.generate_overhead(cam, 200)


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   main()
