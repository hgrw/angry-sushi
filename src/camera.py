from ximea import xiapi
import src.calibration as calibrate
import cv2
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class Camera(object):
    def __init__(self, exposure, dims):

        # Configure camera
        cam = xiapi.Camera(dev_id=0)
        cam.open_device()
        cam.set_exposure(exposure)
        cam.set_imgdataformat('XI_RGB24')
        cam.start_acquisition()

        # Calibration data
        self.targetDimensions = dims
        self.calibrationObjects = []
        self.calibrationParams = {}
        self.undistorted = []
        self.cam = cam
        self.img = xiapi.Image()

    def hardware_white_balance(self):
        self.cam.enable_auto_wb()
        self.cam.get_param(xiapi.XI_PRM_MANUAL_WB, 1)
        self.cam.set_param(xiapi.XI_PRM_MANUAL_WB, 1)

    def software_white_balance(self, img):
        result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result

    def capture_calibration_targets(self, numTargets):

        captured = 0
        while captured < numTargets:
            img = self.stream()
            ret = calibrate.get_points(img, self.targetDimensions)
            if ret is not None:
                cv2.imshow('img', ret.render)
                cv2.waitKey(0)
                self.calibrationObjects.append(ret)
                captured += 1

    def update_exposure(self, exposure):
        self.cam.set_exposure(exposure)

    def get_img(self):
        self.cam.get_image(self.img)
        return self.img.get_image_data_numpy()

    def get_object_hues(self):
        hues = []
        if len(self.calibrationParams['red']) > 1:
            hues.append(np.mean(self.calibrationParams['red'], axis=0).astype(np.uint8))
        if len(self.calibrationParams['green']) > 1:
            hues.append(np.mean(self.calibrationParams['green'], axis=0).astype(np.uint8))
        if len(self.calibrationParams['blue']) > 1:
            hues.append(np.mean(self.calibrationParams['blue'], axis=0).astype(np.uint8))
        if len(self.calibrationParams['yellow']) > 1:
            hues.append(np.mean(self.calibrationParams['yellow'], axis=0).astype(np.uint8))
        return hues

    def stream(self, rectify=False):
        while True:
            if rectify:
                img = calibrate.remove_distortion(self.calibrationParams, self.get_img(), crop=False)
            else:
                img = self.get_img()
            cv2.imshow('img', img)
            k = cv2.waitKey(33)
            if k==99:    # Esc key to stop
                break
            elif k==-1:  # normally -1 returned,so don't print it
                continue
            else:
                print(k) # else print its value
        return img


