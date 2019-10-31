%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute the Gaussian-Newton approximation of the face-to-plane distances
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [b,A] = gn_approx_face2plane(corners, J_corners, ids, planes, weights)
%
b = zeros(7,1);
A = zeros(7,7);
for i = 1 : 4
    id = ids(i);
    c = corners((3*id-2):(3*id));
    J = J_corners((3*id-2):(3*id),:);
    vec_dis = c'*planes(1:3,:) + planes(4,:);
    mat_J = double(J'*planes(1:3,:));
    b = b - sum(vec_dis'.*(weights'*ones(1,7)))';
    A = A + mat_J*sparse(1:length(weights),1:length(weights),weights)*mat_J';
end
b = b/4;
A = A/4;