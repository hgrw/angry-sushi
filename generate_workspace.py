from src.camera import Camera
from src.workspace import Environment
import cv2
import numpy as np
import argparse
import json
import os
import scipy.io as sio


def main():

    # Calibration parameters
    targetDimensions = (6, 9)   # Calibration board dimensions. Required to initialise camera object
    exposure = 70000            # Exposure (gain). Should be kept high to increase depth of field
    bThresh = 95                # Black threshold. Used to detect board
    wThresh = 65                # White theshold. Used to detect goal circle
    hThresh = 11                # Hue theshold. Used to segment coloured blocks
    sThresh = 11                # Saturation threshold. Used to segment coloured blocks
    vThresh = 11                # Value threshold. Used to segment coloured blocks
    pathStep = 0.05             # Pathfinding step size

    worldCorners = [(38, 210),      # Top Left
                    (1140, 25),     # Top Right
                    (1140, 997),    # Bottom Right
                    (38, 809)]      # Bottom Left
    worldLongEdgeMm = 578           # Long edge length of world in mm
    worldShortEdgeMm = 334          # Short edge length of world in mm

    cameraPosition = (1150, 300)    # Vector pointing from workspace origin to camera in top-down rectified frame.
                                    # Used to translate foam object tops along line from top centroid towards camera
                                    # lense and in so doing, find the bottom for each foam object

    # Get image from camera
    cam = Camera(targetDimensions, exposure)

    # Load camera parameters and set white balance from file
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData.json')
    with open(jsonFile, 'r') as fp:
        cam.calibrationParams = json.load(fp)
        cam.set_colour_coefficients()

    # Enter program loop. Breaks only when valid path found
    while True:

        # Get rectified image from virtual overhead camera
        img = cam.get_top_down(cam.get_img(rectify=True, blur=True), worldCorners, worldLongEdgeMm, worldShortEdgeMm)

        # Instantiate environment object
        env = Environment(img, worldCorners)

        # Get workspace objects: blocks faces (tops and sides), cards and goal circle
        env.get_ws_objects(img, cam.get_object_hues(), bThresh, wThresh, hThresh, sThresh, vThresh)

        #cv2.imshow('img', img)
        #cv2.waitKey(1)

        # Get board corners. If corners aren't found, restart loop
        if not env.get_board_corners():
            continue

        # Delete objects that are detected outside the workspace perimeter
        env.update_masks()

        # Use corners to calculate camera extrinsics relative to workspace. Plot origin over image
        env.get_ws_frame(np.asarray(cam.calibrationParams['mtx'], dtype=np.float32),
                         np.asarray(cam.calibrationParams['dist'][0], dtype=np.float32))

        # Convert non-flat objects in workspace to prisms
        env.get_prisms()

        # Invoke the prism base finding method for each prism. Produces junk array (_) as a byproduct. Disregard.
        _ = [prism.bootstrap_base(cameraPosition, env.canvas) for prism in env.prisms]

        # Use prism bottoms to update filled workspace
        env.generate_workspace()

        # Detect start and end points for trajectories
        env.get_start_and_end_points()

        # Show prism geometry on canvas
        env.show_canvas()

        # Check if pathfinding command (s) entered
        k = cv2.waitKey(1)
        
        if k == 115:  # Esc key to stop

            # Commence pathfinding. Restart loop if not valid
            if env.get_paths(pathStep):
                break

        # Pathfinding failed. Purge prisms and restart loop
        env.shapes = []
        env.topSidePairs = []

    # Save data to file and exit
    env.matFile['dimensions'] = env.canvas.shape
    env.matFile['path'] = env.paths
    env.matFile['boardCorners'] = env.boardCorners
    #env.matFile['canvas'] = env.canvas
    #env.matFile['workspace'] = env.workspace
    sio.savemat('/home/mars/git/angry-sushi/angryPath.mat', env.matFile)
    print('PATHING DONE. DATA SAVED TO MAT FILE. PRESS ANY KEY TO EXIT')

    # TODO: call matlab script with submodule

    cv2.waitKey(0)
    exit(0)


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   main()
