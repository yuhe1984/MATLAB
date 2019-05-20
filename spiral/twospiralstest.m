data = load('twospirals.txt');
%data = load('cancer.txt');
x = data(:,1:end-1);
y = data(:,end);
%x = mapminmax(x);
net = newff(x',y',[5,5,5],{'tansig','tansig','tansig','tansig'},'trainrp');
%view(net);
%gscatter(x(:,1),x(:,2),y,'rb','xo');
net.IW{1,1} = (rand(5,2) - 0.5)/2;
net.LW{2} = (rand(5,5) - 0.5)/2;
net.LW{7} = (rand(5,5) - 0.5)/2;
net.LW{12} = (rand(1,5) - 0.5)/2;
net.trainParam.goal = 0.001;
net.divideFcn = '';
net.trainParam.epochs = 15000;
net.trainParam.lr = 0.1;
net.trainParam.deltamax = 0.001;
net.trainParam.mc = 0.9;
net = train(net,x',y');
yy = sim(net,x');
[y,yy'];
x_min = min(x(:, 1)) - 1;
x_max = max(x(:, 1)) + 1;
y_min = min(x(:, 2)) - 1;
y_max = max(x(:, 2)) + 1;
[xx, yy] = meshgrid((x_min:0.02:x_max),(y_min:0.02:y_max));
z = sim(net,[xx(1:end);yy(1:end)]);
m = size(z,2);
for i = 1:m
    if z(i) > 0.5
        z(i) = 1;
    elseif z(i) <= 0.5
        z(i) = 0;
    end
end
z = reshape(z,size(xx));
figure;
contourf(xx, yy, z);
hold on
gscatter(x(:,1),x(:,2),y)