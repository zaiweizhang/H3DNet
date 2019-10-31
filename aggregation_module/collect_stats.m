function [STAT] = collect_stats(PFunc)
% Collect statistics about PFunc
bins_center_vox = zeros(1, 3072);
bins_center_vox2 = zeros(1, 3072);
bins_center_pn = zeros(1, 3072);
bins_center_pn2 = zeros(1, 3072);
bins_corner_vox = zeros(1, 3072);
bins_corner_pn = zeros(1, 3072);
bins_corner_pn2 = zeros(1, 3072);
bins_corner_pn3 = zeros(1, 3072);
bins_face_upper = zeros(1, 3072);
bins_face_lower = zeros(1, 3072);
bins_face_right = zeros(1, 3072);
bins_face_left = zeros(1, 3072);
bins_face_front = zeros(1, 3072);
bins_face_back = zeros(1, 3072);
%
for id = 1 : length(PFunc)
    pfunc = PFunc{id};
    for j = 1 : size(pfunc.gt_objects,1)
        cen = pfunc.gt_objects(j, 1:3)';
        ids = analyze(cen, pfunc.center_vox);
        for k = 1 : length(ids)
            bins_center_vox(ids(k)) = bins_center_vox(ids(k)) + 1;
        end
        ids = analyze(cen, pfunc.center_vox2);
        for k = 1 : length(ids)
            bins_center_vox2(ids(k)) = bins_center_vox2(ids(k)) + 1;
        end
        ids = analyze(cen, pfunc.center_pn);
        for k = 1 : length(ids)
            bins_center_pn(ids(k)) = bins_center_pn(ids(k)) + 1;
        end
        ids = analyze(cen, pfunc.center_pn2);
        for k = 1 : length(ids)
            bins_center_pn2(ids(k)) = bins_center_pn2(ids(k)) + 1;
        end
        % Get corners
        corners = get_corners(pfunc.gt_objects(j,:));
        % Statistics on corners
        for l = 1 : 8
            cor = corners(:, l);
            ids = analyze(cor, pfunc.corner_vox);
            for k = 1 : length(ids)
                bins_corner_vox(ids(k)) = bins_corner_vox(ids(k)) + 1;
            end
            ids = analyze(cor, pfunc.corner_pn);
            for k = 1 : length(ids)
                bins_corner_pn(ids(k)) = bins_corner_pn(ids(k)) + 1;
            end
            ids = analyze(cor, pfunc.corner_pn2);
            for k = 1 : length(ids)
                bins_corner_pn2(ids(k)) = bins_corner_pn2(ids(k)) + 1;
            end
            ids = analyze(cor, pfunc.corner_pn3);
            for k = 1 : length(ids)
                bins_corner_pn3(ids(k)) = bins_corner_pn3(ids(k)) + 1;
            end
        end
        % Statistics on faces
        % upper : [1,2,3,4]
        ids = face_2_plane(corners(:,[1,2,3,4]), pfunc.pred_upper2);
        for k = 1 : length(ids)
            bins_face_upper(ids(k)) = bins_face_upper(ids(k)) + 1;
        end
        % lower : [5,6,7,8]
        ids = face_2_plane(corners(:,[5,6,7,8]), pfunc.pred_lower2);
        for k = 1 : length(ids)
            bins_face_lower(ids(k)) = bins_face_lower(ids(k)) + 1;
        end
        % right : [1,3,5,7]
        ids = face_2_plane(corners(:,[1,3,5,7]), pfunc.pred_right2);
        for k = 1 : length(ids)
            bins_face_right(ids(k)) = bins_face_right(ids(k)) + 1;
        end
        % left : [2,4,6,8]
        ids = face_2_plane(corners(:,[2,4,6,8]), pfunc.pred_left2);
        for k = 1 : length(ids)
            bins_face_left(ids(k)) = bins_face_left(ids(k)) + 1;
        end
        % front : [3,4,7,8]
        ids = face_2_plane(corners(:,[3,4,7,8]), pfunc.pred_front2);
        for k = 1 : length(ids)
            bins_face_front(ids(k)) = bins_face_front(ids(k)) + 1;
        end
        % back : [1,2,5,6]
        ids = face_2_plane(corners(:,[1,2,5,6]), pfunc.pred_back2);
        for k = 1 : length(ids)
            bins_face_back(ids(k)) = bins_face_back(ids(k)) + 1;
        end
    end
end

STAT.bins_center_vox = bins_center_vox;
STAT.bins_center_vox2 = bins_center_vox2;
STAT.bins_center_pn = bins_center_pn;
STAT.bins_center_pn2 = bins_center_pn2;
STAT.bins_corner_vox = bins_corner_vox;
STAT.bins_corner_pn = bins_corner_pn;
STAT.bins_corner_pn2 = bins_corner_pn2;
STAT.bins_corner_pn3 = bins_corner_pn3;
STAT.bins_face_upper = bins_face_upper;
STAT.bins_face_lower = bins_face_lower;
STAT.bins_face_right = bins_face_right;
STAT.bins_face_left = bins_face_left;
STAT.bins_face_front = bins_face_front;
STAT.bins_face_back = bins_face_back;

function [ids] = face_2_plane(corners, planes)
numCorners = size(corners,2);
numPlanes = size(planes, 2);
disMat= zeros(numCorners, numPlanes);
%
for i = 1 : numCorners
    disMat(i,:) = corners(1,i)*planes(1,:)+...
        corners(2,i)*planes(2,:)+...
        corners(3,i)*planes(3,:)+...
        planes(4,:);
end
dif = sqrt(sum(disMat.*disMat)/4);
ids = max(min(3072, floor(dif/0.01)),1);
%

function [corners] = get_corners(object)
%
cen = object(1:3)';
xsize = object(4);
ysize = object(5);
zsize = object(6);
theta = object(7);
nx = [cos(theta), sin(theta), 0]';
ny = [-sin(theta), cos(theta), 0]';
nz = [0,0,1]';
corners(:,1) = cen + nx*xsize/2 + ny*ysize/2 + nz*zsize/2;
corners(:,2) = cen - nx*xsize/2 + ny*ysize/2 + nz*zsize/2;
corners(:,3) = cen + nx*xsize/2 - ny*ysize/2 + nz*zsize/2;
corners(:,4) = cen - nx*xsize/2 - ny*ysize/2 + nz*zsize/2;
corners(:,5) = cen + nx*xsize/2 + ny*ysize/2 - nz*zsize/2;
corners(:,6) = cen - nx*xsize/2 + ny*ysize/2 - nz*zsize/2;
corners(:,7) = cen + nx*xsize/2 - ny*ysize/2 - nz*zsize/2;
corners(:,8) = cen - nx*xsize/2 - ny*ysize/2 - nz*zsize/2;

function [ids] = analyze(query, points)
%
dif = points - query*ones(1, size(points, 2));
dif = sqrt(sum(dif.*dif));
ids = max(min(3072, floor(dif/0.01)),1);
