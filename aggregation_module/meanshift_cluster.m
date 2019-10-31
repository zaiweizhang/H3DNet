function [new_points] = meanshift_cluster(points, sigma)
% Perform mean-shift clustering to obtain the individual clusters from a
% set of points
for iter = 1 : 8
    new_points = points;
    for id = 1 : size(points,2)
        dif = points(:, id)*ones(1, size(points,2)) - points;
        weights = exp(-sum(dif.*dif)/sigma/sigma/2);
        tp = points.*(ones(size(points,1),1)*weights);
        new_points(:, id) = sum(tp')'/sum(weights);
    end
    points = new_points;
end