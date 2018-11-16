function [workspaceImg_TopDown, boundingBox, orientation, realCorners, wsOrigin] = extractWorkspace(img)
    I_gray = rgb2gray(im2double(img));
    level = graythresh(I_gray);

    I = edge(I_gray, 'canny', 0.95); 
    
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

boundingBox = st(areaIndex).BoundingBox;

% figure
% imshow(I);

% rectangle('Position', workspaceBB, 'EdgeColor', 'r', 'LineWidth', 1);
% hold on;

orientation = st(areaIndex).Orientation;

workspaceImg = imcrop(I, boundingBox);

workspaceImg_raw = imcrop(img, boundingBox);

SE = strel('square', 1000);

workspaceImg_Closed = imclose(workspaceImg, SE); %gaps closed in image

stats = regionprops(workspaceImg_Closed, 'BoundingBox', 'Area', ...
    'FilledArea', 'FilledImage', 'Orientation', 'Centroid', 'Perimeter', 'Extrema');

[B,L,N] = bwboundaries(workspaceImg_Closed);

boundary = B{1};
vals = diff(boundary);

xLines = [];

% figure, imshow(workspaceImg_Closed), hold on
for i = 2:length(vals)
    if i < length(vals)
        if vals(i, 2) == vals(i-1, 2) && vals(i, 2) == vals(i+1, 2) && vals(i, 2) ~= 0 
            xPoint = [boundary(i,2), boundary(i,1)];
            xLines = [xLines; xPoint];
            %plot(boundary(i,2), boundary(i,1), 'r*');
        end
    end
end

yLines = [];

for i = 2:length(vals)
    if i < length(vals)
        if vals(i, 1) == vals(i-1, 1) && vals(i, 1) == vals(i+1, 1) && vals(i, 1) ~= 0 
            yPoint = [boundary(i,2), boundary(i,1)];
            yLines = [yLines; yPoint];
        end
    end
end

[minX, I_minX] = min(xLines(:, 1));
[maxX, I_maxX] = max(xLines(:, 1));

[minY, I_minY] = min(yLines(:, 2));
[maxY, I_maxY] = max(yLines(:, 2));

xLines_1 = [];
xLines_2 = [];

for i = 1:length(xLines)
    yThresh = (maxY-minY)/2;
    if xLines(i, 2) < yThresh
        xLines_1 = [xLines_1; [xLines(i, 1), xLines(i, 2)]];
        %plot(xLines(i, 1), xLines(i, 2), 'g*')
    else
        xLines_2 = [xLines_2; [xLines(i, 1), xLines(i, 2)]];
       %  plot(xLines(i, 1), xLines(i, 2), 'r*')
    end
end

[P1, Ind1] = max(xLines_1(:, 1));

P1x = xLines_1(Ind1, 1);
P1y = xLines_1(Ind1, 2);

[P2, Ind2] = max(xLines_2(:, 1));

P2x = xLines_2(Ind2, 1);
P2y = xLines_2(Ind2, 2);

[P3, Ind3] = min(xLines_1(:, 1));

P3x = xLines_1(Ind3, 1);
P3y = xLines_1(Ind3, 2);

[P4, Ind4] = min(xLines_2(:, 1));

P4x = xLines_2(Ind4, 1);
P4y = xLines_2(Ind4, 2);

% plot(P1x, P1y, 'r*');
% plot(P2x, P2y, 'g*');
% plot(P3x, P3y, 'b*');
% plot(P4x, P4y, 'y*');

corners = [P1x, P1y;
            P2x, P2y;
            P3x, P3y;
            P4x, P4y;];
        
wsOrigin = [P3x, P3y];

% for i = 1:length(xLines_1)
%     P2x = xLines_1(i, 1);
%     P2y = xLines_1(i, 2);
%     plot(P2x, P2y, 'r*');
% end

% for i = 1:length(corners)
%     plot(corners(i, 1), corners(i, 2), 'r*')
% end

A3_aspectRatio = 297/420;

realCornersX = [P3x P3x P1x P1x];
heightY = A3_aspectRatio * abs(P3x-P1x);
realCornersY = [P3y P3y + heightY P3y P3y+heightY];

realCorners = [P3x P3y; P3x P3y + heightY;
               P1x P3y; P1x P3y + heightY];

% figure, imshow(workspaceImg), hold on
% plot(corners(:, 1), corners(:, 2), 'b*');
% plot(realCorners(:, 1), realCorners(:, 2), 'r*');

tform = fitgeotrans(corners,realCorners,'projective');

workspaceImg2 = imwarp(workspaceImg_raw, tform, 'OutputView', imref2d(size(workspaceImg)*1));

wsRect = [P3x P3y abs(P1x-P3x) heightY];

% figure, imshow(workspaceImg2), hold on

%plot(realCornersX, realCornersY, 'r*');
% rectangle('Position', wsRect, 'EdgeColor', 'r', 'LineWidth', 3);

workspaceImg_TopDown = imcrop(workspaceImg2, wsRect);

end