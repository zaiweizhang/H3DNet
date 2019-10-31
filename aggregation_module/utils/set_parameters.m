function [Para_local] = set_parameters()
%
Para_local.sigma_face_back = 0.1;
Para_local.sigma_face_front = 0.1;
Para_local.sigma_face_left = 0.1;
Para_local.sigma_face_right = 0.1;
Para_local.sigma_face_lower = 0.1;
Para_local.sigma_face_upper = 0.1;
Para_local.sigma_corner_pn =  0.15;
Para_local.sigma_corner_vox = 0.15;
Para_local.sigma_center_pn = 0.15;
Para_local.sigma_center_vox = 0.1;
Para_local.sigma_center_vox2 = 0.1;
%
Para_local.lambda_center_vox2 = 1;
Para_local.lambda_center_vox = 1;
Para_local.lambda_center_pn = 1;
Para_local.lambda_corner_vox = 0.25;
Para_local.lambda_corner_pn =  0.25;
Para_local.lambda_face_back =  0.0625;
Para_local.lambda_face_front = 0.0625;
Para_local.lambda_face_left = 0.0625;
Para_local.lambda_face_right = 0.0625;
Para_local.lambda_face_lower = 0.0625;
Para_local.lambda_face_upper = 0.0625;