import src.calibration as calibrate
import src.filter_tools as filter
import src.plot_tools as plot
import src.calibration as cal
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
    targetDimensions = (6, 9)   # Calibration board dimensions. Required to initialise camera object
    exposure = 60000            # Exposure (gain). Should be kept high to increase depth of field
    bThresh = 70                # Black threshold. Used to detect board
    wThresh = 65                # White theshold. Used to detect goal circle
    hThresh = 10                # Hue theshold. Used to segment coloured blocks
    sThresh = 10                # Saturation threshold. Used to segment coloured blocks
    vThresh = 10                # Value threshold. Used to segment coloured blocks
    testStart = (355, 355)
    testEnd = (677, 677)
    pathStep = 0.075            # Pathfinding step size
    pathing = False             # Set to false while workspace generation completes. When true, pathfinding commences

    blockout = [(1140, 0),      # Blockout region for bottom of camera mount and arm mount
                (1140, 1023),
                (200, 960),
                (1080, 960)]

    # Get image from camera
    cam = Camera(targetDimensions, exposure)
    env = Environment()

    # Load camera parameters
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData_AEV.json')
    print("LOADING CAMERA PARAMETERS")
    with open(jsonFile, 'r') as fp:
        cam.calibrationParams = json.load(fp)
        cam.set_colour_coefficients()
        cam.get_rectify_mask(blockout)

    #cam.calibrate_lens()
    #cam.record_video('/home/mars/Videos/nightrider_local.avi')

    while True:
        # Get rectified image from camera
        img = cam.get_img(rectify=True, blur=True)

        # Canvas to plot computer vision and pathing output for visualisation
        canvas = img.copy()

        # Extract board, shapes and tops
        #env.boardMask, sidesMask, topsMask, env.sides, env.tops = filter.get_elements(canvas,
        #                                                         cam.get_object_hues(),
        #                                                         cam.rectifyMask,
        #                                                         bThresh, hThresh, sThresh, vThresh)

        env.get_workspace_objects(img, cam.get_object_hues(), cam.rectifyMask,
                                  bThresh,
                                  wThresh,
                                  hThresh,
                                  sThresh,
                                  vThresh)

        #bw = plot.show_mask(plot.show_mask(img, blacks, 2), whites, 1)

        # Get board corners
        env.get_board_corners()
        if len(env.boardCorners) is 4:

            # Plot corners
            for point in env.boardCorners:
                cv2.circle(canvas, point, 4, [0, 0, 255], 3)

            # Use corners to calculate camera extrinsics relative to workspace. Plot origin over image
            canvas = env.get_workspace_frame(canvas,
                                             np.asarray(cam.calibrationParams['mtx'], dtype=np.float32),
                                             np.asarray(cam.calibrationParams['dist'][0], dtype=np.float32))

            # Using the workspace corners, a filled workspace mask can be generated and then used to update masks to
            # improve accuracy
            env.update_masks()


        # Plot shape tops, board bask and blockout zones on image
        canvas = plot.show_mask(plot.show_mask(plot.show_mask(canvas, env.boardMask, 2), env.sides, 1), env.tops, 0)
        cv2.line(canvas, blockout[0], blockout[1], [0, 255, 0], 3)
        cv2.line(canvas, blockout[2], blockout[3], [0, 255, 0], 3)

        # Generate topdown view of workspace
        topDown = env.get_top_down(canvas)


        # Plot corners for workspace origin frame
        #for point in env.wsOrigin:
        #    cv2.circle(img, (int(point[0]), int(point[1])), 4, [255, 0, 0], 3)

        #cv2.circle(img, testStart, 10, [255, 0, 0])
        #cv2.circle(img, testEnd, 10, [255, 0, 0])

        # Fill gaps in board
        #env.fill_board()

        # Get board corners
        #env.get_board_corners(np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        #filter.get_box(filter.get_edges(env.boardFilled), img)


        if pathing:
            canvas = env.get_top_down(canvas)
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
        #cv2.imshow("bw", ~cv2.cvtColor(cam.get_img(rectify=True), cv2.COLOR_BGR2GRAY) & ~cam.rectifyMask)

        k = cv2.waitKey(1)
        if k == 115:    # Esc key to stop
            print("PATHING MODE ENGAGED")
            pathing = True
    exit(0)

    # Generate top-down view of image set


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   main()
