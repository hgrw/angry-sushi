from src.file_tools import Images
import src.calibration as cal
import src.filter_tools as filter
import src.plot_tools as plot
import src.math_tools as utils
import cv2
import numpy as np
import argparse

def main(imagePath, mode):

    # Load image data for task (mode)
    imageSet = Images(imagePath, mode)

    if mode[0:11] == 'calibration':

        ######################
        #                    #
        # PART A STARTS HERE #
        #                    #
        ######################

        # dimensions for calibration1,2,3 target
        dimensions = (6, 8)
        warped = False

        if mode == 'calibrationWarped':
            dimensions = (7, 13)
            warped = True

        # Assign image set to object
        obj = imageSet.__getattribute__(mode)

        # Collect points for calibration targest
        cal.generate_calibration(obj, dimensions, updatePoints=warped)

        # Generate camera model
        cal.calibrate_camera(obj)

        # Print camera calibration matrix, intrinsics, extrinsics
        cal.print_calibration_matrix(obj, 6.2, 5)

        # Remove distortion on image set
        cal.remove_distortion(obj, crop=False, showError=True)

        if warped:

            # Show undistorted image set
            cv2.imshow('rectified',
                       plot.view_set(imageSet.calibrationWarped.undistorted, (1280, 1024)))
        else:

            # Generate top-down view of image set
            cal.generate_overhead(obj, 200)

            # Show top-down view of dataset
            cv2.imshow('rectified',
                       plot.view_set(obj.rectified, (1280, 1024)))

        cv2.waitKey(0)

    else:

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

   parser = argparse.ArgumentParser(description="METR4202 Image Classification by Harry Roache-Wilson")
   parser.add_argument("path", type=str, help="path to image directory")
   parser.add_argument("mode", type=str, help="image dataset: [calibration1 | calibration2 | calibration3 | \
                                                calibrationWarped | basic | skilled1 | skilled2]")
   args = parser.parse_args()

   main(args.path, args.mode)
