% Estimate normals for a ply file. 

ptCloud = pcread('plyfiles/scene0001_00_vh_clean_2.ply');
% % ptCloud = pcread('npyfiles/scene0004_00_vert.ply');
% pointscolor=uint8(zeros(ptCloud.Count,3));
% pointscolor(:,1)=255;
% pointscolor(:,2)=255;
% pointscolor(:,3)=51;ptCloud.Color=pointscolor;
normals = pcnormals(ptCloud, 'k',100);
disp(ptCloud.Count);
figure
pcshow(ptCloud)
set(gcf,'color','w');
set(gca,'color','w');

title('Estimated Normals of Point Cloud')
hold on
x = ptCloud.Location(1:10:end,1);
y = ptCloud.Location(1:10:end,2);
z = ptCloud.Location(1:10:end,3);
u = normals(1:10:end,1);
v = normals(1:10:end,2);
w = normals(1:10:end,3);quiver3(x,y,z,u,v,w);
hold off

sensorCenter = [0,-0.3,0.3]; 
for k = 1 : numel(x)
   p1 = sensorCenter - [x(k),y(k),z(k)];
   p2 = [u(k),v(k),w(k)];
   % Flip the normal vector if it is not pointing towards the sensor.
   angle = atan2(norm(cross(p1,p2)),p1*p2');
   if angle > pi/2 || angle < -pi/2
       u(k) = -u(k);
       v(k) = -v(k);
       w(k) = -w(k);
   end
end
figure
pcshow(ptCloud)
set(gcf,'color','w');
set(gca,'color','w');

title('Adjusted Normals of Point Cloud')
hold on
quiver3(x, y, z, u, v, w);
hold off

