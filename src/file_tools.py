import glob
import sys

from src.features import ImageSet

image_list = []



class Images(object):

    """
    Images Contained in Class

    calibration1
    calibration2
    calibration3
    calibrationWarped
    basic
    skilled1
    skilled2


    """

    def __init__(self, imageDirectory, modeCmd):
        self._extended = imageDirectory + "METR4202 Extended Test Set/"
        self._standard = imageDirectory + "PS2 Images/"
        self.calibration1 = []
        self.calibration2 = []
        self.calibration3 = []
        self.calibrationWarped = []
        self.basic = []
        self.skilled1 = []
        self.skilled2 = []

        sys.stdout.write('Loading Images')
        sys.stdout.flush()
        if modeCmd == 'calibration1':
            self.calibration1 = ImageSet([f for f in glob.glob(self._standard + "Calibration - One/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'calibration2':
            self.calibration2 = ImageSet([f for f in glob.glob(self._standard + "Calibration - Two/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'calibration3':
            self.calibration3 = ImageSet([f for f in glob.glob(self._extended + "Calibration - Set 3/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'calibrationWarped':
            self.calibrationWarped = ImageSet([f for f in glob.glob(self._extended + "Warped Calibration/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'basic':
            self.basic = ImageSet([f for f in glob.glob(self._standard + "Basic/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'skilled1':
            self.skilled1 = ImageSet([f for f in glob.glob(self._standard + "Skillful/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        if modeCmd == 'skilled2':
            self.skilled2 = ImageSet([f for f in glob.glob(self._extended + "Advanced - Set 2/*.png")])
            sys.stdout.write('.')
            sys.stdout.flush()

        sys.stdout.write('.\n')
        sys.stdout.flush()
