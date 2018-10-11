import cv2

class ImageSet(object):

    def __init__(self, paths):
        self.paths = paths
        self.images = [cv2.imread(f) for f in paths]
        self.imageWithBoxes = []
        self.undistorted = []
        self.rectified = []
        self.calibrationSet = []
        self.calibrationParams = {}
        self.edges = []
        self.boxes = []
        self.lines = []
        self.len = len(paths)

