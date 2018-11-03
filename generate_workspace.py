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
    targetDimensions = (6, 9)
    #exposure = 10000
    exposure = 60000
    hThresh = 10
    sThresh = 10
    vThresh = 10
    #bThresh = 70
    bThresh = 70
    testStart = (355, 355)
    testEnd = (677, 677)
    pathStep = 0.075
    pathing = False

    blockout = [(1140, 0), (1140, 1023),
            (200, 960), (1080, 960)]        # Blockout region for bottom of camera mount and arm mount

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

    #while True:
    #    img = cam.get_img()
        """
        The below function provides matrices that map 3d points into the world frame. When calibrating, the first calibration
        target is used to define the world frame, and so must be aligned with the camera mount and/or manipulator mount.
        
        Using these matrices, we can generate a virtual camera at a very specific location. This won't give us a rectangular workspace!
        
        In theory, using the virtual-camera-rectified image, in conjunction with an image of a chocked up workspace, the findhomography function
        in cv2 should give us the euler angles of the workspace.
        
        """
    #    objPoints, corners, mtx, dist, rvecs, tvecs = cam.get_world_frame_data(0)

        # Find the rotation and translation vectors.
        #_, rvecs, tvecs, inliers = cv2.solvePnPRansac(objPoints, corners, mtx, dist)

    #    img = plot.render_origin_frame(img, corners, rvecs, tvecs, mtx, dist)
    #    cv2.imshow('img', img)
    #    cv2.waitKey(0)

    #cv2.imshow('nadir', cal.get_nadir(cam.get_img(rectify=True), cam.calibrationParams['mtx']))
    #cv2.waitKey(0)
    #exit(0)
    #cam.calibrate_lens()
    #cam.record_video('/home/mars/Videos/nightrider_local.avi')

    while True:
        #img = calibrate.get_nadir(cam.get_img(rectify=True))
        img = cam.get_img(rectify=True)
        # Extract elements by colour
        env.boardMask, sidesMask, topsMask, env.sides, env.tops = filter.get_elements(img.copy(),
                                                                 cam.get_object_hues(),
                                                                 cam.rectifyMask,
                                                                 bThresh, hThresh, sThresh, vThresh)


        #env.boardMask = env.boardMask & filter.remove_components(
        #    cv2.dilate(env.boardMask, np.ones((15, 15), np.uint8), iterations=3), largest=True)

        env.get_board_corners()
        if len(env.boardCorners) is 4:

            # Plot corners
            for point in env.boardCorners:
                cv2.circle(img, point, 4, [0, 0, 255], 3)

            # Get camera intrinsics
            mtx = np.asarray(cam.calibrationParams['mtx'], dtype=np.float32)
            dist = np.asarray(cam.calibrationParams['dist'][0], dtype=np.float32)

            # Workspace corners in workspace frame [0, 1]
            objp = np.zeros((4, 3), np.float32)
            objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2)

            # SolvePnPRansac wants a very specific matrix dimension for corners. Manipulate corner matrix to conform.
            # Specifically, it is expecting a square with same dimensions as checkerboard square. Since we know board
            # dimensions, we can generate such a square on which our origin lays. Then it must be manipulated so that
            # it conforms to the pedantic cv2 specification. To be fair, it because cv2 is so well optimised.
            env.wsOrigin = cal.generate_origin_square(env.boardCorners)
            numpyCorners = np.expand_dims(np.asarray([np.asarray(cnr, dtype=np.float32).T for cnr in env.wsOrigin]),
                                          axis=1)

            # Plot corners for workspace origin square
            for point in env.wsOrigin:
                cv2.circle(img, (int(point[0]), int(point[1])), 4, [255, 0, 0], 3)

            # Generate rotation and translation vectors for calibration target
            _, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, numpyCorners, mtx, dist)

            try:
                img = plot.render_origin_frame(img, numpyCorners, rvecs, tvecs, mtx, dist)
            except OverflowError:
                print('No line to draw')
        #cv2.circle(img, testStart, 10, [255, 0, 0])
        #cv2.circle(img, testEnd, 10, [255, 0, 0])

        # Fill gaps in board
        #env.fill_board()

        # Get board corners
        #env.get_board_corners(np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        #filter.get_box(filter.get_edges(env.boardFilled), img)

        # Plot shape tops, board bask and blockout zones on image
        canvas = plot.show_mask(plot.show_mask(img, env.boardMask, 2), topsMask, 1)
        cv2.line(canvas, blockout[0], blockout[1], [0, 255, 0], 3)
        cv2.line(canvas, blockout[2], blockout[3], [0, 255, 0], 3)

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
