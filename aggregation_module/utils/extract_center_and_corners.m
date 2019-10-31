
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Used in deriving the corners and the center from the parameters of the
% object
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [cen, corners] = extract_center_and_corners(cur_object)
%
cen = cur_object(1:3)';
xsize = cur_object(4);
ysize = cur_object(5);
zsize = cur_object(6);
theta = cur_object(7);
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
