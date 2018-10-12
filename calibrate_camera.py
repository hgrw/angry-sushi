import src.calibration as cal
import src.filter_tools as filter
import src.plot_tools as plot
import src.math_tools as utils
import cv2
import numpy as np
import argparse
import json
import os

from src.camera import Camera
from src.camera import NumpyEncoder

def main(exposure):

    # Calibration parameters
    targetDimensions = (6, 9)
    numTargets = 10

    # Get image from camera
    cam = Camera(exposure, targetDimensions)

    # Collect points for calibration target
    cam.capture_calibration_targets(numTargets)

    # Generate camera model
    cal.calibrate_camera(cam, targetDimensions)

    # Print camera calibration matrix, intrinsics, extrinsics
    cal.print_calibration_matrix(cam, 6.2, 5)

    cam.stream(rectify=True)

    # Save camera parameters
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData.json')
    print "CALIBRATION COMPLETE, SAVING CAMERA PARAMETERS to : ", jsonFile
    with open(jsonFile, 'w') as fp:
        json.dump(cam.calibrationParams, fp, cls=NumpyEncoder)

    exit(0)

    # Generate top-down view of image set
    cal.generate_overhead(cam, 200)

    # Show top-down view of dataset
    cv2.imshow('rectified',
               plot.view_set(calibrationTargets.rectified, (1280, 1024)))

    cv2.waitKey(0)


    ######################
    #                    #
    # PART B STARTS HERE #
    #                    #
    ######################

    # Assign image set to object
    obj = imageSet.__getattribute__(mode)

    if mode == 'skilled2':
        obj.images = [filter.trim_images(img) for img in obj.images]

    # Calculated edges
    obj.edges = [filter.get_edges(img) for img in obj.images]

    # Infill components
    obj.edges = [filter.infill_components(img) for img in obj.edges]

    # Close
    obj.edges = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5))) for img in obj.edges]

    # Remove small components
    obj.edges = [np.asarray(filter.remove_components(img, minSize=10000), dtype=np.uint8) for img in obj.edges]

    # Simple Canny edges
    obj.edges = [cv2.Canny(img, 40, 40, 20, L2gradient=True) for img in obj.edges]

    # Generate bouding boxes
    obj.boxes = [filter.get_boxes(edge) for edge in obj.edges]

    obj.imagesWithBoxes = [utils.draw_bounding_boxes(obj.images[i], obj.boxes[i]) for i in range(0, len(obj.images))]

    cv2.imshow('bounding boxes',
               plot.view_set(obj.imagesWithBoxes, (1280, 1024)))
    cv2.waitKey(0)


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   parser.add_argument("exposure", type=int, help="Exposure value. High value is good for tight aperture, deep field")
   args = parser.parse_args()
   main(args.exposure)
