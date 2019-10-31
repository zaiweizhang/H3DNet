function [new_object] = perturb_object(org_object)

new_object = org_object;
new_object(1:6) = new_object(1:6) + (rand(1,6)-0.5)*0.4;
new_object(4:6) = abs(new_object(4:6));

end