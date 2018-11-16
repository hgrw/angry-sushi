function [pathX_f, pathY_f] = getPath()

% Access an image acquisition device
vidobj = videoinput('gentl', 1, 'BGRA8Packed');
src=getselectedsource(vidobj);
src.AEAGEnable = 'True';
% List the video input object's configurable properties.vidobj.FramesPerTrigger = 50;
% Open the preview window

img3 = getsnapshot(vidobj);

imshow(img3);

[workspaceImg, BB, O, C, wsOrigin] = extractWorkspace(img3);

I_gray = rgb2gray(im2double(workspaceImg));
level = graythresh(I_gray);

I = edge(I_gray, 'canny', 0.3*level); 

SE = strel('square', 50);

I2 = imclose(I, SE);

stats = regionprops(I, 'BoundingBox', 'Area', ...
    'FilledArea', 'FilledImage', 'Orientation', 'Centroid', 'Perimeter', 'Extrema');

% figure, imshow(I2), hold on
% for i = 1:length(stats)
%     BB = stats(i).BoundingBox;
%     rectangle('Position', BB, 'EdgeColor', 'r', 'LineWidth', 3);
% end

I3 = bwareaopen(I2,300);

% figure, imshow(I3), hold on

[B,L,N] = bwboundaries(I3, 'noholes');

for i = 1:length(B)
    boundary = B{i};
    %plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
end

obj1 = B{2};
obj2 = B{3};

obj1_aspectRatio = (max(obj1(:, 1)) - min(obj1(:, 1))) / ...
                   (max(obj1(:, 2)) - min(obj1(:, 2)));
               
obj2_aspectRatio = (max(obj2(:, 1)) - min(obj2(:, 1))) / ...
                   (max(obj2(:, 2)) - min(obj2(:, 2)));
               
goal = [];
target = [];      

if abs(1-obj1_aspectRatio) < abs(1-obj2_aspectRatio)
    goal = obj1;
    target = obj2;
else
    goal = obj2;
    target = obj1;
end

goalCentroidY = max(goal(:, 1)) - (max(goal(:, 1))-min(goal(:, 1)))/2;
goalCentroidX = max(goal(:, 2)) - (max(goal(:, 2))-min(goal(:, 2)))/2;
            


minDist = [inf, inf];

for i = 1:length(target)
    Px = target(i, 2);
    Py = target(i, 1);
    d = sqrt((Px - goalCentroidX)^2 - (Py - goalCentroidY)^2);
    if d < min(minDist(1))
        minDist = [d, i];
    end
end

targetX = target(i, 2);
targetY = target(i, 1);



distX = targetX - goalCentroidX;
distY = targetY - goalCentroidY;

n_Steps = 0;

if distX > distY
    n_Steps = int16(abs(distX/40));
else
    n_Steps = int16(abs(distY/40));
end

xIncrement = distX/n_Steps;
yIncrement = distY/n_Steps;

pathX = [];
pathY = [];

for i = 1:n_Steps
    pathX = [pathX targetX - i*xIncrement];
    pathY = [pathY targetY - i*yIncrement];
end

[sizeY, sizeX] = size(I);

scalingX = 420/sizeX;
scalingY = 297/sizeY;

offsetX = 200;
offsetY = 270;

figure
subplot(1, 2, 1)
imshow(I)
subplot(1, 2, 2)
imshow(I2)

figure
imshow(I3), hold on
plot(targetX, targetY, 'r*');
plot(goalCentroidX, goalCentroidY, 'r*');
for i = 1:length(pathX)
    plot(pathX(i), pathY(i), 'r*');
end

pathX_f = double(pathX * scalingX);
pathY_f = double(pathY * scalingY);





