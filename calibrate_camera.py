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


hueLocs = []
calibrating = True


def click_and_crop(event, x, y, flags, param):

    # grab references to the global variables
    global hueLocs, calibrating

    # if the left button is clicked, calibration routine underway
    if event == cv2.EVENT_LBUTTONDOWN:
        hueLocs.append([x, y])
        print('collected {} hues'.format(len(hueLocs)))

    if event == cv2.EVENT_RBUTTONDOWN:
        calibrating = False


def generate_baselines(cam, hues, message):
    print(message)
    global calibrating
    hsvCalibrationValues = []

    while calibrating:
        img = cam.get_img(blur=True)
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('img', click_and_crop)
        cv2.imshow('img', img)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for hue in hues:
        hsvCalibrationValues.append(hsv[hue[1], hue[0], :])
    return hsvCalibrationValues, []


def main():

    # Calibration parameters
    targetDimensions = (6, 9)               # Calibration target dimensions in squares
    exposure = 70000                        # Exposure (gain). Should be high to increase depth of field
    numTargets = 8                          # Number of calibration targets to collect

    worldCorners = [(38, 210),      # Top Left
                    (1140, 25),     # Top Right
                    (1140, 997),    # Bottom Right
                    (38, 809)]      # Bottom Left

    # Used in the mouse-click callback function
    global hueLocs, calibrating

    # Instantiate camera
    cam = Camera(targetDimensions, exposure)

    # Store white balance coefficients
    cam.white_balance()

    # Set focus and exposure
    cam.calibrate_lens()

    # Collect points for calibration target
    cam.capture_calibration_targets(numTargets)

    # Generate camera model
    cal.calibrate_camera(cam, targetDimensions)

    # Show blockout regions for aligning workspace
    cal.align_tilt(cam, worldCorners)

    # Generate values for the tops of all objects to be incorporated into work space
    cam.calibrationParams['red'], hueLocs = generate_baselines(cam, hueLocs, "SELECT RED TOPS")
    calibrating = True
    cam.calibrationParams['green'], hueLocs = generate_baselines(cam, hueLocs, "SELECT GREEN TOPS")
    calibrating = True
    cam.calibrationParams['blue'], hueLocs = generate_baselines(cam, hueLocs, "SELECT BLUE TOPS")
    calibrating = True
    cam.calibrationParams['yellow'], hueLocs = generate_baselines(cam, hueLocs, "SELECT YELLOW TOPS")
    calibrating = True
    cam.calibrationParams['purple'], hueLocs = generate_baselines(cam, hueLocs, "SELECT CARD BACKS")

    #cam.stream(rectify=True)

    # Save camera parameters
    jsonFile = os.path.join(os.path.dirname(__file__), 'cameraData.json')
    print "CALIBRATION COMPLETE, SAVING CAMERA PARAMETERS to : ", jsonFile
    with open(jsonFile, 'w') as fp:
        json.dump(cam.calibrationParams, fp, cls=NumpyEncoder)

    exit(0)


if __name__ == "__main__":

   parser = argparse.ArgumentParser(description="METR4202 imaging and pathing project Harry Roache-Wilson")
   main()
