function [opt_object] = object_opt_v1(PFunc, init_object, Para)
%
opt_object = init_object;
for outer = 1 : 8
    WEIGHT = cue_reweighting(PFunc, opt_object, Para);
    %WEIGHT = cue_reweighting_withsem(PFunc, opt_object, Para);
    WEIGHT = adjust_weights(WEIGHT, Para);
    for inner = 1 : 20
        [para_diff] = gaussian_newton_step(PFunc, opt_object, WEIGHT);
        e_cur = energy_object2cues(PFunc, WEIGHT, opt_object);
        alpha = 1;
        flag = 0;
        for searchID = 1 : 10
            next_object = opt_object;
            next_object(1:7) = opt_object(1:7) + para_diff*alpha;
            e_next = energy_object2cues(PFunc, WEIGHT, next_object);
            if e_next < e_cur
                opt_object = next_object;
                flag = 1;
                break;
            end
            alpha = alpha/2;
        end
        if flag == 0
            break;
        end
    end
end
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [para_diff] = gaussian_newton_step(PFunc, cur_object, Weight)
%
cen = cur_object(1:3)';
% Extract the first order approximation of the corners


A = zeros(7,7);
b = zeros(7,1);

% center alignment term (vox resolution 1)
dif = PFunc.center_vox - cen*ones(1, size(PFunc.center_vox,2));
A(1:3,1:3) = A(1:3,1:3) + sum(Weight.vec_center_vox)*eye(3);
b(1:3) = b(1:3) + sum(dif'.*kron(Weight.vec_center_vox', ones(1,3)))';
% center alignment term (vox resolution 1)
dif = PFunc.center_vox2 - cen*ones(1, size(PFunc.center_vox2,2));
A(1:3,1:3) = A(1:3,1:3) + sum(Weight.vec_center_vox2)*eye(3);
b(1:3) = b(1:3) + sum(dif'.*kron(Weight.vec_center_vox2', ones(1,3)))';
% center alignment term (pn)
dif = PFunc.center_pn - cen*ones(1, size(PFunc.center_pn,2));
A(1:3,1:3) = A(1:3,1:3) + sum(Weight.vec_center_pn)*eye(3);
b(1:3) = b(1:3) + sum(dif'.*kron(Weight.vec_center_pn', ones(1,3)))';
% Corner alignment term
[corners, J_corners] = Jacobi_corners(cur_object);
%
for id = 1 : 8
    J = J_corners((3*id-2):(3*id),:);
    dif = PFunc.corner_vox - corners((3*id-2):(3*id))*ones(1, size(PFunc.corner_vox,2));
    vec_w = Weight.vec_corner_vox(id,:);
    A = A + (J'*J)*sum(vec_w);
    b = b + sum((dif'*J).*(vec_w'*ones(1,7)))';
    dif = PFunc.corner_pn - corners((3*id-2):(3*id))*ones(1, size(PFunc.corner_pn,2));
    vec_w = Weight.vec_corner_pn(id,:);
    A = A + (J'*J)*sum(vec_w);
    b = b + sum((dif'*J).*(vec_w'*ones(1,7)))';
end
% upper : [1,2,3,4]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [1,2,3,4],...
    PFunc.pred_upper2, Weight.vec_face_upper);
b = b + b_dif;
A = A + A_dif;
% lower : [5,6,7,8]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [5,6,7,8],...
    PFunc.pred_lower2, Weight.vec_face_lower);
b = b + b_dif;
A = A + A_dif;
% right : [1,3,5,7]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [1,3,5,7],...
    PFunc.pred_right2, Weight.vec_face_right);
b = b + b_dif;
A = A + A_dif;
% left : [2,4,6,8]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [2,4,6,8],...
    PFunc.pred_left2, Weight.vec_face_left);
b = b + b_dif;
A = A + A_dif;
% front : [3,4,7,8]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [3,4,7,8],...
    PFunc.pred_front2, Weight.vec_face_front);
b = b + b_dif;
A = A + A_dif;
% back : [1,2,5,6]
[b_dif, A_dif] = gn_approx_face2plane(corners, J_corners, [1,2,5,6],...
    PFunc.pred_back2, Weight.vec_face_back);
b = b + b_dif;
A = A + A_dif;
eigVals = eig(A)';
if min(eigVals) < max(eigVals)/10
    A = A + max(eigVals)*eye(7)/10;
end
para_diff = (A\b)';








