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
    def __init__(self, dims, exposure):

        # Configure camera. Default exposure value of 50k
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

    def calibrate_lens(self):

        print("SET FOCUS. PRESS e TO UPDATE EXPOSURE. ESC ONCE DONE.")

        h, w, _ = self.get_img().shape
        while True:
            frame = self.get_img()
            zoomed = cv2.resize(frame[int(h/2) - 15: int(h/2) + 15, int(w/2) - 15: int(w/2) + 15],
                                (300, 300), 0, 0, cv2.INTER_NEAREST)
            frame[int(h/2) - 150 : int(h/2) + 150, int(w/2) - 150 : int(w/2) + 150, :] = zoomed
            cv2.imshow('video', frame)
            k = cv2.waitKey(1)
            if k == 27:    # Esc key to stop
                break
            if k == 101:
                exposure = input("ENTER EXPOSURE")
                print("SETTING EXPOSURE TO: {}".format(exposure))
                self.update_exposure(int(exposure))
        cv2.destroyAllWindows()

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

    def get_img(self, rectify=False):
        self.cam.get_image(self.img)
        if not rectify:
            return self.img.get_image_data_numpy()
        else:
            return calibrate.remove_distortion(self.calibrationParams, self.img.get_image_data_numpy(), crop=False)

    def record_video(self, output):

        frame = self.get_img()
        height, width, channels = frame.shape
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(output, fourcc, 30.0, (width, height))
        print("RECORDING VIDEO. PRESS ESC TO STOP")
        while True:
            frame = self.get_img()

            out.write(np.asarray(frame, dtype=np.uint8))
            cv2.imshow('video', frame)
            k = cv2.waitKey(1)
            if k == 27:    # Esc key to stop
                break
        out.release()
        cv2.destroyAllWindows()

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
        if len(self.calibrationParams['purple']) > 1:
            hues.append(np.mean(self.calibrationParams['purple'], axis=0).astype(np.uint8))
        return hues

    def stream(self, rectify=False):
        while True:
            if rectify:
                img = calibrate.remove_distortion(self.calibrationParams, self.get_img(), crop=False)
            else:
                img = self.get_img()
            cv2.imshow('img', img)
            k = cv2.waitKey(33)
            if k == 27:    # Esc key to stop
                break
        return img


