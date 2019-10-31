%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [e] = energy_object2cues(PFunc, Weight, cur_object)
% Compute the current energy of the objective function
[cen, corners] = extract_center_and_corners(cur_object);
%
dif = PFunc.center_vox - cen*ones(1, size(PFunc.center_vox,2));
e = sum(sum(dif.*dif).*Weight.vec_center_vox);
dif = PFunc.center_vox2 - cen*ones(1, size(PFunc.center_vox2,2));
e = e + sum(sum(dif.*dif).*Weight.vec_center_vox2);
dif = PFunc.center_pn - cen*ones(1, size(PFunc.center_pn,2));
e = e + sum(sum(dif.*dif).*Weight.vec_center_pn);
for id = 1 : 8
    dif = PFunc.corner_vox - corners(:,id)*ones(1, size(PFunc.corner_vox,2));
    e = e + sum(sum(dif.*dif).*Weight.vec_corner_vox(id,:));
    dif = PFunc.corner_pn - corners(:,id)*ones(1, size(PFunc.corner_pn,2));
    e = e + sum(sum(dif.*dif).*Weight.vec_corner_pn(id,:));
end
% upper : [1,2,3,4]
sqrDis = face_2_plane_sqrDis(corners(:,[1,2,3,4]), PFunc.pred_upper2);
e = e + sum(sqrDis.*Weight.vec_face_upper);
% lower : [5,6,7,8]
sqrDis = face_2_plane_sqrDis(corners(:,[5,6,7,8]), PFunc.pred_lower2);
e = e + sum(sqrDis.*Weight.vec_face_lower);
% right : [1,3,5,7]
sqrDis = face_2_plane_sqrDis(corners(:,[1,3,5,7]), PFunc.pred_right2);
e = e + sum(sqrDis.*Weight.vec_face_right);
% left : [2,4,6,8]
sqrDis = face_2_plane_sqrDis(corners(:,[2,4,6,8]), PFunc.pred_left2);
e = e + sum(sqrDis.*Weight.vec_face_left);
% front : [3,4,7,8]
sqrDis = face_2_plane_sqrDis(corners(:,[3,4,7,8]), PFunc.pred_front2);
e = e + sum(sqrDis.*Weight.vec_face_front);
% back : [1,2,5,6]
sqrDis = face_2_plane_sqrDis(corners(:,[1,2,5,6]), PFunc.pred_back2);
e = e + sum(sqrDis.*Weight.vec_face_back);

