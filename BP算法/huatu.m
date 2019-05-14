%data = load('twospirals.txt');
%data = load('LR-testSet2.txt');
%data = load('yihuo.txt');
global prob;
x = prob.inputData;
y = prob.outputData;
x_min = min(x(:, 1)) - 1;
x_max = max(x(:, 1)) + 1;
y_min = min(x(:, 2)) - 1;
y_max = max(x(:, 2)) + 1;
[xx, yy] = meshgrid((x_min:0.02:x_max),(y_min:0.02:y_max));
z = predict([xx(1:end);yy(1:end)]');
z = reshape(z,size(xx));
figure;
contourf(xx, yy, z);
hold on
gscatter(x(:,1),x(:,2),y)