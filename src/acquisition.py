from ximea import xiapi
import cv2

class Camera(object):
  
  def __init__(self, exposure):
    self.exposure = exposure

    # Configure camera
    cam = xiapi.Camera(dev_id=0)
    cam.open_device()
    cam.set_exposure(exposure)
    cam.set_imgdataformat('XI_RGB24')
    cam.start_acquisition()
    self.cam = cam
    self.img = xiapi.Image()
  
  def get_img(self):
    self.cam.get_image(self.img)
    return self.img.get_image_data_numpy()

cam = Camera(10000)
while True:
	cv2.imshow('img', cam.get_img())
	k = cv2.waitKey(33)
	if k==27:    # Esc key to stop
		break
	elif k==-1:  # normally -1 returned,so don't print it
		continue
	else:
		print k # else print its value 

