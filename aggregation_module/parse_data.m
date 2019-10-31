function [Data] = parse_data(folder_gt, folder_pred)
% Read the data from zaiwei
temp = dir(folder_gt);
for id = 1 : (length(temp)-2)
    if id == 149 || id == 798 
        continue;
    end
    name = temp(id+2).name;
    scene_name = name(1:12);
    load([folder_gt, name]);
    clusters = extract_isolated(gtr, 0.0001);
    Data{id}.scene_name = scene_name;
    Data{id}.gt_objects = clusters;
    % Parse plane
    
    % Parse point
    %   Parse center
    path_name = [folder_pred, 'point/', scene_name, '_point_objcue.mat'];
    load(path_name);
    ids = find(subpc_mask);
    gt_center = gt_center(ids,:)';
    Data{id}.gt_center = extract_isolated2(gt_center, 1e-4);
    Data{id}.pred_center = pred_center';
    % Parse semantic 
    gt_sem = gt_sem(ids);
    Data{id}.gt_sem = gt_sem;
    Data{id}.pred_sem = pred_sem;
    %   Parse corner
    gt_corner = gt_corner(ids,:)';
    Data{id}.gt_corner = extract_isolated2(gt_corner, 1e-4);
    Data{id}.pred_corner = pred_corner';
    % Parse plane
    path_name = [folder_pred, 'plane/', scene_name, '_plane_objcue.mat'];
    load(path_name);
    Data{id}.gt_back = extract_isolated2(gt_back', 1e-4);
    Data{id}.gt_front = extract_isolated2(gt_front', 1e-4);
    Data{id}.gt_left = extract_isolated2(gt_left', 1e-4);
    Data{id}.gt_right = extract_isolated2(gt_right', 1e-4);
    Data{id}.gt_lower = extract_isolated2(gt_lower', 1e-4);
    Data{id}.gt_upper = extract_isolated2(gt_upper', 1e-4);
    Data{id}.pred_back = pred_back';
    Data{id}.pred_front = pred_front';
    Data{id}.pred_left = pred_left';
    Data{id}.pred_right = pred_right';
    Data{id}.pred_lower = pred_lower';
    Data{id}.pred_upper = pred_upper';
    % Parse voxel
    %   Parse corner
    path_name = [folder_pred, 'voxel/', scene_name, '_corner_0.06_vox.mat'];
    load(path_name);
    Data{id}.corner_vox = corner_vox';
    path_name = [folder_pred, 'voxel/', scene_name, '_center_0.06_vox.mat'];
    load(path_name);
    Data{id}.center_vox = center_vox';
    path_name = [folder_pred, 'voxel/', scene_name, '_center_0.06_vox.mat'];
    load(path_name);
    Data{id}.center_vox2 = center_vox';
    %   Parse center
    fprintf('%d\n', id);
end

function [clusters] = extract_isolated2(data, error)
%
clusters = data(:,1);
for id = 2 : size(data,2)
    query = data(:,id);
    flag = 1;
    for j = 1 : size(clusters,2)
        dif = norm(clusters(:,j)-query);
        if dif < error
            flag = 0;
        end
    end
    if flag == 1
        clusters = [clusters,query];
    end
end
tp = sum(clusters.*clusters);
ids = find(tp > 0.0001);
clusters = clusters(:,ids);


function [clusters] = extract_isolated(data, error)
%
clusters = data(1, :);
for id = 2 : size(data,1)
    query = data(id, :);
    flag = 1;
    for j = 1 : size(clusters,1)
        dif = norm(clusters(j,:)-query);
        if dif < error
            flag = 0;
        end
    end
    if flag == 1
        clusters = [clusters;query];
    end
end
tp = clusters(:, 1:7)';
tp = sum(tp.*tp);
ids = find(tp > 0.01);
clusters = clusters(ids,:);
