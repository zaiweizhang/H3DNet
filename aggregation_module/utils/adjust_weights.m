%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Balance based on the length of the vectors
function [Weight_out] = adjust_weights(Weight, Para)
Weight.vec_center_vox = Para.lambda_center_vox*Weight.vec_center_vox/length(Weight.vec_center_vox);
Weight.vec_center_vox2 = Para.lambda_center_vox2*Weight.vec_center_vox2/length(Weight.vec_center_vox2);
Weight.vec_center_pn = Para.lambda_center_pn*Weight.vec_center_pn/length(Weight.vec_center_pn);
Weight.vec_corner_vox = Para.lambda_corner_vox*Weight.vec_corner_vox/size(Weight.vec_corner_vox,2);
Weight.vec_corner_pn = Para.lambda_corner_pn*Weight.vec_corner_pn/size(Weight.vec_corner_pn,2);
Weight.vec_face_upper = Para.lambda_face_upper*Weight.vec_face_upper/length(Weight.vec_face_upper);
Weight.vec_face_lower = Para.lambda_face_lower*Weight.vec_face_lower/length(Weight.vec_face_lower);
Weight.vec_face_left = Para.lambda_face_left*Weight.vec_face_left/length(Weight.vec_face_left);
Weight.vec_face_right = Para.lambda_face_right*Weight.vec_face_right/length(Weight.vec_face_right);
Weight.vec_face_front = Para.lambda_face_front*Weight.vec_face_front/length(Weight.vec_face_front);
Weight.vec_face_back = Para.lambda_face_back*Weight.vec_face_back/length(Weight.vec_face_back);
Weight_out = Weight;