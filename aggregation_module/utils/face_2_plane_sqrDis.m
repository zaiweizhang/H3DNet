%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Used in evaluating the face compatibility score
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sqrDis] = face_2_plane_sqrDis(corners, planes)
numPlanes = size(planes, 2);
disMat= zeros(4, numPlanes);
%
for i = 1 : 4
    disMat(i,:) = corners(1,i)*planes(1,:)+...
        corners(2,i)*planes(2,:)+...
        corners(3,i)*planes(3,:)+...
        planes(4,:);
end
sqrDis = sum(disMat.*disMat)/4;