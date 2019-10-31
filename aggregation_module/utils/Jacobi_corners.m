%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute the Jacobi of the object corners
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [corners, J_corners] = Jacobi_corners(cur_object)
%
cen = cur_object(1:3)';
xsize = cur_object(4);
ysize = cur_object(5);
zsize = cur_object(6);
theta = cur_object(7);
nx = [cos(theta), sin(theta), 0]';
ny = [-sin(theta), cos(theta), 0]';
nz = [0,0,1]';
%
signs = zeros(8,3);
signs(1,:) = [1,1,1];
signs(2,:) = [-1,1,1];
signs(3,:) = [1,-1,1];
signs(4,:) = [-1,-1,1];
signs(5,:) = [1,1,-1];
signs(6,:) = [-1,1,-1];
signs(7,:) = [1,-1,-1];
signs(8,:) = [-1,-1,-1];
%
corners = zeros(24,1);
J_corners = zeros(24,7);
for i = 1 : 8
    corners((3*i-2):(3*i)) = cen +...
        signs(i,1)*nx*xsize/2 +...
        signs(i,2)*ny*ysize/2 +...
        signs(i,3)*nz*zsize/2;
    J_corners((3*i-2):(3*i),1:3) = eye(3);
    J_corners((3*i-2):(3*i),4) = signs(i,1)*nx/2;
    J_corners((3*i-2):(3*i),5) = signs(i,2)*ny/2;
    J_corners((3*i-2):(3*i),6) = signs(i,3)*nz/2;
    J_corners((3*i-2):(3*i),7) = -(signs(i,1)*xsize*ny+signs(i,2)*ysize*nx)/2;
end
