% % Access an image acquisition device
% vidobj = videoinput('gentl', 1, 'BGRA8Packed');
% src=getselectedsource(vidobj);
% src.AEAGEnable = 'True';
% % List the video input object's configurable properties.vidobj.FramesPerTrigger = 50;
% % Open the preview window
% 
% img = getsnapshot(vidobj);

%imshow(img);

 I_gray = rgb2gray(im2double(img));
 level = graythresh(I_gray);

 I = edge(I_gray, 'canny', 0.9);   
 
%  imshow(I);


st = regionprops(I, 'BoundingBox', 'Area', 'FilledArea', 'FilledImage', 'Orientation');

boundAreas = [];

% imshow(I);
% hold on
for i = 1:length(st)
    BB = st(i).BoundingBox;
    area = BB(3)*BB(4);
    boundAreas = [boundAreas, area];
    %rectangle('Position', BB, 'EdgeColor', 'r', 'LineWidth', 1);
end

[maxArea, areaIndex] = max(boundAreas);

workspaceBB = st(areaIndex).BoundingBox;

% figure
%imshow(I);

% rectangle('Position', workspaceBB, 'EdgeColor', 'r', 'LineWidth', 1);
% hold on;

workspaceO = st(areaIndex).Orientation;

workspaceImg = imcrop(I, workspaceBB);

workspaceImg_raw = imcrop(img, workspaceBB);

workspaceLoc = [workspaceBB(1) + workspaceBB(3)/2, workspaceBB(2) + workspaceBB(4)/2];

SE = strel('square', 1000); %Square morphology of 10 pixels to close the image gaps

workspaceImg_rot = imrotate(workspaceImg, workspaceO); 
workspaceImg_rot_raw = imrotate(workspaceImg_raw, workspaceO);
%imshow(workspaceImg_rot_raw);
workspaceImg_rot2 = imclose(workspaceImg_rot, SE); %gaps closed in image 
workspaceImg_rot2_raw = workspaceImg_rot_raw;
 
[H,T,R] = hough(workspaceImg_rot2);

P  = houghpeaks(H,5,'threshold',ceil(0.1*max(H(:))));

lines = houghlines(I,T,R,P,'FillGap',200,'MinLength',20);


stats = regionprops(workspaceImg_rot2, 'BoundingBox', 'Area', ...
    'FilledArea', 'FilledImage', 'Orientation', 'Centroid', 'Perimeter', 'Extrema');

lines = houghlines(workspaceImg_rot2,T,R,P,'FillGap',5,'MinLength',7);


[B,L,N] = bwboundaries(workspaceImg_rot2);


% figure, imshow(workspaceImg_rot2), hold on
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'r', 'LineWidth', 2)
% end

% figure, imshow(workspaceImg_rot2), hold on
% for i = 1:length(stats.Extrema)
%     BB = stats(i).BoundingBox;
%     px = stats.Extrema(i, 1)
%     py = stats.Extrema(i, 2)
%     area = BB(3)*BB(4);
%     boundAreas = [boundAreas, area];
%     plot(px, py, 'r*');
% end

boundary = B{1};
vals = diff(boundary);
% 
% figure, imshow(workspaceImg_rot2), hold on
% plot(stats.Centroid(1), stats.Centroid(2), 'r*')
% hold on

c_x = stats.Centroid(1);
c_y = stats.Centroid(2);

D = [];

for i = 1:length(boundary)
    b_x = boundary(i, 2);
    b_y = boundary(i, 1);
    d_x = abs(b_x - c_x);
    d_y = abs(b_y - c_y);
    d_r = sqrt(b_x - c_x)^2;
    d = [d_x, d_y];
    D = [D; d];
end

maxD = max(D);

% plot(-maxD(1) + c_x, -maxD(2) + c_y, 'r*');

% C = corner(workspaceImg_rot2, 4);
% figure, imshow(workspaceImg_rot2), hold on
% plot(C(:,1),C(:,2),'r*');

% figure, imshow(workspaceImg_rot2), hold on

xLines = [];

for i = 2:length(vals)
    if i < length(vals)
        if vals(i, 2) == vals(i-1, 2) && vals(i, 2) == vals(i+1, 2) && vals(i, 2) ~= 0 
%             plot(boundary(i,2), boundary(i,1), 'r*');
            xPoint = [boundary(i,2), boundary(i,1)];
            xLines = [xLines; xPoint];
        end
    end
end

yLines = [];

for i = 2:length(vals)
    if i < length(vals)
        if vals(i, 1) == vals(i-1, 1) && vals(i, 1) == vals(i+1, 1) && vals(i, 1) ~= 0 
%             plot(boundary(i,2), boundary(i,1), 'r*');
            yPoint = [boundary(i,2), boundary(i,1)];
            yLines = [yLines; yPoint];
        end
    end
end

[minX, I_minX] = min(xLines(:, 1));
[maxX, I_maxX] = max(xLines(:, 1));

[minY, I_minY] = min(yLines(:, 2));
[maxY, I_maxY] = max(yLines(:, 2));


    
% plot(xLines(I_minX, 1), xLines(I_minX, 2), 'r*'); 
% plot(xLines(I_maxX, 1), xLines(I_maxX, 2), 'r*'); 
% 
% 
% plot(yLines(I_minY, 1), yLines(I_minY, 2), 'r*'); 
% plot(yLines(I_maxY, 1), yLines(I_maxY, 2), 'r*'); 

% for i = 1:length(xLines)
%     plot(xLines(i, 1), xLines(i, 2), 'r*');
% end
% for i = 1:length(yLines)
%     plot(yLines(i, 1), yLines(i, 2), 'g*');
% end

xLines_1 = [];
xLines_2 = [];

figure, imshow(workspaceImg_rot2), hold on
for i = 1:length(xLines)
    yThresh = (maxY-minY)/2;
    if xLines(i, 2) < yThresh
        xLines_1 = [xLines_1; [xLines(i, 1), xLines(i, 2)]];
    else
        xLines_2 = [xLines_2; [xLines(i, 1), xLines(i, 2)]];
    end
end

C = [min(xLines_1(:, 1)), min(xLines_1(:, 2));
    min(xLines_2(:, 1)), min(xLines_2(:, 2));
max(xLines_1(:, 1)), max(xLines_1(:, 2));
max(xLines_2(:, 1)), max(xLines_2(:, 2))];

for i = 1:length(C)
    plot(C(i, 1), C(i, 2), 'r*')
end





% SE2 = strel('square', 5);
% 
% IM2 = imdilate(workspaceImg_rot,SE2);
% imshow(IM2);

% C = corner(IM2);
% hold on
% plot(C(:,1),C(:,2),'r*');