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
    exposure = 50000            # Exposure (gain). Should be kept high to increase depth of field
    bThresh = 70                # Black threshold. Used to detect board
    wThresh = 65                # White theshold. Used to detect goal circle
    hThresh = 10                # Hue theshold. Used to segment coloured blocks
    sThresh = 10                # Saturation threshold. Used to segment coloured blocks
    vThresh = 10                # Value threshold. Used to segment coloured blocks
    testStart = (355, 355)
    testEnd = (677, 677)
    pathStep = 0.15             # Pathfinding step size
    pathing = False             # Set to false while workspace generation completes. When true, pathfinding commences

    blockout = [(1140, 36), (1140, 1000),   # Right border
                (187, 860), (1140, 1000),   # Bottom border
                (400, 892), (400, 932),     # Actuator mounting marker 1
                (600, 920), (600, 960),     # Actuator mounting marker 2
                (800, 950), (800, 990)]     # Actuator mounting marker 3

    cameraPosition = (130, 130) # Vector pointing from workspace origin to camera in top-down rectified frame. Used to
                                # translate foam object tops along line from top centroid towards camera lense and in
                                # so doing, find the bottom for each foam object

    # Get image from camera
    cam = Camera(targetDimensions, exposure)

    # Load camera parameters
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData_AEV.json')
    print("LOADING CAMERA PARAMETERS")
    with open(jsonFile, 'r') as fp:
        cam.calibrationParams = json.load(fp)
        cam.set_colour_coefficients()
        cam.get_rectify_mask(blockout)

    #cam.calibrate_lens()

    # Enter program loop. Breaks only when valid path found
    while True:

        # Get rectified image from camera
        img = cam.get_img(rectify=True, blur=True)

        # Instantiate environment object
        env = Environment(img.copy(), blockout)

        # Get workspace objects: blocks faces (tops and sides), cards and goal circle
        env.get_ws_objects(img, cam.get_object_hues(), cam.rectifyMask, bThresh, wThresh, hThresh, sThresh, vThresh)

        # Get board corners. If corners aren't found, restart loop
        if not env.get_board_corners():
            continue

        # Use corners to calculate camera extrinsics relative to workspace. Plot origin over image
        env.get_ws_frame(np.asarray(cam.calibrationParams['mtx'], dtype=np.float32),
                         np.asarray(cam.calibrationParams['dist'][0], dtype=np.float32))

        # Delete objects that are detected outside the workspace perimeter
        env.update_masks()

        # Convert objects to top down view
        env.get_top_down()

        # Convert non-flat objects in workspace to prisms
        env.get_prisms()

        # Invoke the prism base finding method for each prism. Produces junk array (_) as a byproduct. Disregard.
        _ = [prism.bootstrap_base(cameraPosition) for prism in env.prisms]

        # Use prism bottoms to update filled workspace
        workspace = env.generate_workspace()

        # Un-extrude tops. We can do this by translating each top along a line pointing back to camera in overhead view
        # and taking the point along the line that has the most pixels where the top intersect the edge.

        # Detect start and end points for trajectories
        #env.get_start_and_end_points()

        # Enter path generation mode
        if pathing:
            path = tree.generate_path(env.boardMask, testStart, testEnd, pathStep)
            if path is not None:
                env.canvasTd = plot.plot_path(env.canvas, path)
            cv2.imshow("canvas", env.canvasTd)
        else:
            #cv2.imshow("canvas", plot.show_mask(plot.show_mask(env.canvas, env.boardMask, 2), env.sides, 1))
            cv2.imshow("canvas", plot.show_mask(env.canvasTd, workspace, 2))

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


        # Create a
        #env.shapeMask = filter.get_shapes(img, env.worldMask ^ env.boardMask)
        #board = shapes ^ ~env.boardMask

        #edges = filter.infill_components(workSpace)

        #img = filter.get_edges(img, saturationThreshold)
        #img = filter.get_clahe(img)
        #saturationMask = filter.color_mask(img, saturationThreshold)

        #img = filter.remove_components()
        #cv2.imshow("Contours", plot.view_pair(env.boardMask, env.shapeMask))
        #cv2.imshow("bw", ~cv2.cvtColor(cam.get_img(rectify=True), cv2.COLOR_BGR2GRAY) & ~cam.rectifyMask)

        # Remove shapes and top-side pairs array
        env.shapes = []
        env.topSidePairs = []
        k = cv2.waitKey(1)
        if k == 115:    # Esc key to stop
            print("PATHING MODE ENGAGED")
            pathing = True
    exit(0)

    # Generate top-down view of image set


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   main()
