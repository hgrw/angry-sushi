% Access an image acquisition device
vidobj = videoinput('gentl', 1, 'BGRA8Packed');
src=getselectedsource(vidobj);
src.AEAGEnable = 'True';
% List the video input object's configurable properties.vidobj.FramesPerTrigger = 50;
% Open the preview window

img3 = getsnapshot(vidobj);

imshow(img3);

