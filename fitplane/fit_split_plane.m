% First use RANSAC to detect planes, then use K nearest neighbors to build a graph for every plane   (connect point to its neighbors). Then split connected components to split planes. 
% You need to specify the path to read ply file. 
% Two parameters: K: number of nearest neighbor. min_npt: after we split planes, if point number of that plane > min_npt, we keep this plane. 

ptCloud = pcread('plyfiles/scene0001_00_vh_clean_2.ply');
pcshow(ptCloud);
set(gcf,'color','w');
set(gca,'color','w');
K = 100;
min_npt = 300;
out_name = sprintf('K%d-MinN%d',K,min_npt);

new_ids  = get_split_planes(ptCloud, K, min_npt);
write_ply(ptCloud, new_ids, out_name);

function new_ids = get_split_planes(ptCloud, K, min_npt)
% [params, ids, new_ptCloud] = get_planes(ptCloud);
[params, ids,] = get_planes2(ptCloud);
num_pt = ptCloud.Count();
% num_pt = new_ptCloud.Count();
% disp(size(params));
% disp(size(ids));
ids = reshape(ids, [num_pt, 1]);
new_ids = -1*ones(size(ids));
% disp(size(ids));
s = size(params);
num_planes = s(1);
c=0;
for i= 1:num_planes
    InlierPtcloud = select(ptCloud, find(ids==i));
    ids1 = find(ids==i);
%     figure('Name', 'Plane');
%     pcshow(InlierPtcloud);
    locations = InlierPtcloud.Location();
    [num_inpt,t] = size(locations);

    split_ids = knn_split(locations, K);
    split_ids =reshape(split_ids, [num_inpt, 1]);
    for s_id=1:size(unique(split_ids))
        s_pt = select(InlierPtcloud, find(split_ids==s_id));
        ids2 = find(split_ids==s_id);
        if s_pt.Count()>min_npt
            c=c+1;
            new_ids(ids1(ids2)) = c;
%             figure('Name', sprintf('plane-%d-%d',i,s_id));
%             pcshow(s_pt);
        end 
    end
end
end

function [params, ids] = get_planes2(ptCloud)
maxDistance = 0.05;
% referenceVector = [0,0,1];
maxAngularDistance = 5;
num_pt=ptCloud.Count();
remainPtCloud = ptCloud;
params = zeros(1,5);
ids = -ones(num_pt, 1);
count = -1;

for i=1:30
    disp('start plane');
    disp(i);
%     if rem(i,3)==1
%         referenceVector = [0,0,1];
%     elseif rem(i, 3)==2
%         referenceVector = [0,1,0];
%     else
%         referenceVector = [1,0,0];
%     end
    [model1,inlierIndices,outlierIndices] = pcfitplane(remainPtCloud, maxDistance);
    plane1 = select(remainPtCloud,inlierIndices);
    remainPtCloud = select(remainPtCloud,outlierIndices);
    s1 = size(outlierIndices);
    if s1(1)<100
        disp(i)
        break
    end
    s = size(inlierIndices);
    if s(1)>300 
        count = count+1;
        p = [model1.Parameters, count];
        params = [params; p];
        n_plane_pt = plane1.Count();
        for j=1:n_plane_pt
            r = sum(abs(ptCloud.Location()-plane1.Location(j,:)), 2);
            idx = find(r<1e-4);
            ids(idx())=count;
        end

%         if count==6
%             figure('Name', 'Plane');
%             pcshow(plane1);
%         end
    end
end
% params = params(2:end,:);
% disp(size(params));
% disp(size(ids));

% fileID = fopen('scene0001_00_2_params.txt','w');
% fprintf(fileID,'%.5f %.5f %.5f %.5f %d\n',params);
% fclose(fileID);
% % % 
% fileID = fopen('scene0001_00_2_plane_ids.txt','w');
% fprintf(fileID,'%d\n',ids);
% fclose(fileID);
end


function split_ids = knn_split(locations, K)
[num_pt,t] = size(locations);
A = zeros(num_pt, num_pt);

KNN_ids = knnsearch(locations, locations, 'K', K);
% disp(KNN_ids);
disp('finish knn');
for i=1:num_pt
    A(i,KNN_ids(i,:)) = 1.0;
end
% G is not symmetric so it's not a adjacency matrix
A2 = transpose(A);
if 1==1
    % option1: intersection 
    A = A2.*A;
else
    % option2: union
    A = A2+A-A2.*A;
end
% disp(A);
G = graph(A);
disp('finish building graph');
split_ids = conncomp(G);
% disp(unique(split_ids));
disp('finish split')
end
function write_ply(ptCloud, new_ids, filename)
xyz = ptCloud.Location();
npt = ptCloud.Count();
[n_planes,t] = size(unique(new_ids));
colors = zeros(npt, 3);
n_planes = n_planes-1;

color_plane = zeros(n_planes, 3);

for i=1:n_planes
    color_plane(i,:) = rand(1,3); 
end
for i=1:npt
    if new_ids(i)>0
        colors(i,:) = color_plane(new_ids(i),:);

    end
end
new_ptCloud = pointCloud(xyz,'Color', colors);
pcwrite(new_ptCloud, filename);
end 
