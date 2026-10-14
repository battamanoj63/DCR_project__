clc;
clear;
close all;

%% =========================================================
% STAGE 2 (ENHANCED) - SAFE, CURRENT-AWARE A* PATH PLANNING
% Autonomous USV - Emergency Assistance
%
% ORIGINAL STAGE 2 + CUSTOM KNN ML
%
% ML addition:
%   1. Generate mathematical/synthetic collision-risk samples
%   2. Train custom KNN manually
%   3. Test KNN accuracy
%   4. Save knn_risk_model.mat
%
% No Statistics and Machine Learning Toolbox required.
% =========================================================

disp('==============================================');
disp(' STAGE 2 ENHANCED - CURRENT-AWARE SAFE A* + ML');
disp('==============================================');

%% =========================================================
% 1. SETTINGS
% ==========================================================

map_size      = 100;
safety_margin = 4;
start         = [10 10];
target        = [90 90];

u_nom         = 1.2;
planning_sea  = 'moderate';

comfort_dist  = 10;
w_clear       = 3;

%% =========================================================
% 2. MAP, SAFETY MAP, SEA
% ==========================================================

[map,safe_map] = make_map(safety_margin);

safe_map(start(2),start(1))   = 0;
safe_map(target(2),target(1)) = 0;

sea = make_sea(planning_sea,1,0.1,1);

%% =========================================================
% 3. DRIFT FIELD (current + wind + wave drift) AT t = 0
% ==========================================================

[gx,gy] = meshgrid(1:map_size,1:map_size);

[cu,cv] = sea_current(sea,gx,gy,0);
[wx0,wy0] = sea_wind(sea,0);

drift_u = cu + wx0 + sea.drift(1);
drift_v = cv + wy0 + sea.drift(2);

v_drift_max = max(hypot(drift_u(:),drift_v(:)));

%% =========================================================
% 4. CLEARANCE PENALTY MAP
% ==========================================================

[oy,ox] = find(map == 1);

clear_d = zeros(map_size,map_size);

for yy = 1:map_size
    for xx = 1:map_size
        clear_d(yy,xx) = min(hypot(ox-xx,oy-yy));
    end
end

penalty = max(0,(comfort_dist-clear_d)/comfort_dist).^2;

%% =========================================================
% 5. A* SEARCH
% ==========================================================

neighbors = [-1 -1; 0 -1; 1 -1; ...
             -1  0;        1  0; ...
             -1  1; 0  1; 1  1];

gCost  = inf(map_size,map_size);
fOpen  = inf(map_size,map_size);

closed = false(map_size,map_size);

parX = zeros(map_size,map_size);
parY = zeros(map_size,map_size);

h = @(x,y) hypot(x-target(1),y-target(2)) / ...
           (u_nom+v_drift_max);

gCost(start(2),start(1)) = 0;
fOpen(start(2),start(1)) = h(start(1),start(2));

found = false;
expansions = 0;

tic;

while true

    [fbest,lin] = min(fOpen(:));

    if isinf(fbest)
        break;
    end

    [cy,cx] = ind2sub(size(fOpen),lin);

    if cx == target(1) && cy == target(2)

        found = true;
        break;

    end

    fOpen(cy,cx) = inf;
    closed(cy,cx) = true;
    expansions = expansions + 1;

    for n = 1:size(neighbors,1)

        dx = neighbors(n,1);
        dy = neighbors(n,2);

        nx = cx + dx;
        ny = cy + dy;

        if nx < 1 || nx > map_size || ...
           ny < 1 || ny > map_size

            continue;

        end

        if safe_map(ny,nx) == 1 || closed(ny,nx)

            continue;

        end

        % No diagonal corner cutting
        if dx ~= 0 && dy ~= 0

            if safe_map(cy,cx+dx) == 1 || ...
               safe_map(cy+dy,cx) == 1

                continue;

            end

        end

        len = hypot(dx,dy);

        % Ground speed
        gs = u_nom + ...
            (drift_u(ny,nx)*dx + ...
             drift_v(ny,nx)*dy)/len;

        gs = max(gs,0.3);

        stepCost = ...
            len/gs + ...
            w_clear*penalty(ny,nx)*len;

        newG = gCost(cy,cx) + stepCost;

        if newG < gCost(ny,nx)

            gCost(ny,nx) = newG;

            parX(ny,nx) = cx;
            parY(ny,nx) = cy;

            fOpen(ny,nx) = ...
                newG + h(nx,ny);

        end

    end

end

search_time = toc;

if ~found
    error('No safe path could be found.');
end

fprintf('\nA* found a safe path (%d expansions, %.2f s).\n', ...
    expansions,search_time);

%% =========================================================
% 6. RECONSTRUCT RAW PATH
% ==========================================================

astar_raw = target;
cur = target;

while ~isequal(cur,start)

    p = [parX(cur(2),cur(1)) ...
         parY(cur(2),cur(1))];

    if p(1) == 0
        error('Path reconstruction failed.');
    end

    cur = p;

    astar_raw = [cur; astar_raw]; %#ok<AGROW>

end

%% =========================================================
% 7. SMOOTH THE PATH
% ==========================================================

% 7a. Line-of-sight shortcuts

keep = 1;
i = 1;
nr = size(astar_raw,1);

while i < nr

    j = nr;

    while j > i+1 && ...
            ~los_free(safe_map,astar_raw(i,:),astar_raw(j,:))

        j = j-1;

    end

    keep(end+1) = j; %#ok<SAGROW>

    i = j;

end

waypoints = astar_raw(keep,:);

% 7b. Re-sample every 1 m

s = [0; cumsum(hypot( ...
    diff(waypoints(:,1)), ...
    diff(waypoints(:,2))))];

si = unique([(0:1:s(end))'; s(end)]);

dense = [ ...
    interp1(s,waypoints(:,1),si), ...
    interp1(s,waypoints(:,2),si)];

% 7c. Round corners

path = dense;

for win = [11 9 7 5 3]

    cand = dense;

    cand(:,1) = movmean(dense(:,1),win);
    cand(:,2) = movmean(dense(:,2),win);

    cand(1,:) = start;
    cand(end,:) = target;

    idx = sub2ind( ...
        size(safe_map), ...
        round(cand(:,2)), ...
        round(cand(:,1)));

    if all(safe_map(idx) == 0)

        path = cand;
        break;

    end

end

%% =========================================================
% 8. METRICS
% ==========================================================

len_raw = sum(hypot( ...
    diff(astar_raw(:,1)), ...
    diff(astar_raw(:,2))));

len_smooth = sum(hypot( ...
    diff(path(:,1)), ...
    diff(path(:,2))));

idx = sub2ind( ...
    size(clear_d), ...
    round(path(:,2)), ...
    round(path(:,1)));

min_clear = min(clear_d(idx));

fprintf('\n==============================================\n');
fprintf(' A* PATH PLANNING RESULTS\n');
fprintf('==============================================\n');
fprintf('Raw A* path length      : %.2f m (%d points)\n', ...
    len_raw,size(astar_raw,1));

fprintf('Smoothed path length    : %.2f m (%d points)\n', ...
    len_smooth,size(path,1));

fprintf('Estimated travel time   : %.1f s at %.1f m/s\n', ...
    len_smooth/u_nom,u_nom);

fprintf('Closest obstacle approach: %.2f cells (hard margin %d)\n', ...
    min_clear,safety_margin);

fprintf('Planning sea state      : %s\n',sea.name);
fprintf('==============================================\n');

%% =========================================================
% 9. SAVE ORIGINAL A* DATA
% ==========================================================

save('astar_path.mat','path','astar_raw','map','safe_map', ...
     'start','target','safety_margin');

disp(' ');
disp('Saved astar_path.mat');

%% =========================================================
% 10. CUSTOM KNN ML
% =========================================================
%
% ONLY ADDITION TO ORIGINAL STAGE 2
%
% Features:
% 1. Distance
% 2. Closing Speed
% 3. Relative Angle
% 4. Wind Speed
% 5. Current Speed
% 6. Heading Error
% 7. TTC
%
% Output:
% 0 = SAFE
% 1 = COLLISION RISK
%
% Training data are mathematically generated.
% ==========================================================

disp(' ');
disp('==============================================');
disp(' CUSTOM KNN ML TRAINING');
disp('==============================================');

rng(100);

numSamples = 4000;

% Feature 1
distance = 2 + 28*rand(numSamples,1);

% Feature 2
closingSpeed = -1 + 3*rand(numSamples,1);

% Feature 3
relativeAngle = -180 + 360*rand(numSamples,1);

% Feature 4
windSpeed = 0.2 + 5*rand(numSamples,1);

% Feature 5
currentSpeed = 0.1 + 2.5*rand(numSamples,1);

% Feature 6
headingError = -180 + 360*rand(numSamples,1);

% Feature 7 - TTC
ttc = zeros(numSamples,1);

for i = 1:numSamples

    if closingSpeed(i) > 0.05

        ttc(i) = distance(i)/closingSpeed(i);

    else

        ttc(i) = 60;

    end

end

ttc = min(ttc,60);

%% Mathematical collision-risk rule

riskRule = ...
    (distance < 7) | ...
    ((ttc < 5) & (closingSpeed > 0.3)) | ...
    ((distance < 11) & ...
     (abs(relativeAngle) < 25) & ...
     (closingSpeed > 0.8));

Y = double(riskRule);

%% Feature matrix

X = [ ...
    distance ...
    closingSpeed ...
    relativeAngle ...
    windSpeed ...
    currentSpeed ...
    headingError ...
    ttc];

%% =========================================================
% TRAIN / TEST SPLIT
% ==========================================================

rng(101);

order = randperm(numSamples);

nTrain = round(0.80*numSamples);

trainIdx = order(1:nTrain);
testIdx = order(nTrain+1:end);

Xtrain = X(trainIdx,:);
Ytrain = Y(trainIdx);

Xtest = X(testIdx,:);
Ytest = Y(testIdx);

%% =========================================================
% NORMALIZATION
% ==========================================================

mu = mean(Xtrain,1);

sigma = std(Xtrain,0,1);

sigma(sigma < 1e-9) = 1;

XtrainN = ...
    (Xtrain-mu)./sigma;

XtestN = ...
    (Xtest-mu)./sigma;

%% =========================================================
% CUSTOM KNN CLASSIFIER
% ==========================================================

K = 9;

predictedTest = zeros(size(Ytest));

for i = 1:size(XtestN,1)

    distances = sqrt( ...
        sum((XtrainN-XtestN(i,:)).^2,2));

    [~,idx] = sort(distances,'ascend');

    kk = min(K,length(idx));

    nearest = idx(1:kk);

    probability = ...
        mean(Ytrain(nearest));

    predictedTest(i) = ...
        probability >= 0.5;

end

%% Accuracy

accuracy = ...
    mean(predictedTest == Ytest)*100;

fprintf('Training samples : %d\n',nTrain);
fprintf('Testing samples  : %d\n',length(Ytest));
fprintf('K value          : %d\n',K);
fprintf('KNN accuracy     : %.2f %%\n',accuracy);

%% Confusion counts

TP = sum((predictedTest == 1) & (Ytest == 1));
TN = sum((predictedTest == 0) & (Ytest == 0));
FP = sum((predictedTest == 1) & (Ytest == 0));
FN = sum((predictedTest == 0) & (Ytest == 1));

fprintf('\nKNN confusion counts:\n');
fprintf('TP = %d\n',TP);
fprintf('TN = %d\n',TN);
fprintf('FP = %d\n',FP);
fprintf('FN = %d\n',FN);

%% =========================================================
% SAVE KNN MODEL
% ==========================================================

knnModel.Xtrain = XtrainN;
knnModel.Ytrain = Ytrain;

knnModel.mu = mu;
knnModel.sigma = sigma;

knnModel.K = K;

knnModel.featureNames = { ...
    'Distance'
    'ClosingSpeed'
    'RelativeAngle'
    'WindSpeed'
    'CurrentSpeed'
    'HeadingError'
    'TTC'};

knnModel.accuracy = accuracy;

save('knn_risk_model.mat','knnModel');

disp(' ');
disp('Saved knn_risk_model.mat');

%% =========================================================
% 11. FIGURE 1 - MAP, CURRENT FIELD, PATHS
% ==========================================================

figure('Name','Stage 2 - Path planning');

imagesc(map + 0.35*safe_map.*(1-map));

set(gca,'YDir','normal');

colormap(gca,[1 1 1; 0.85 0.85 0.85; 0.4 0.4 0.4]);

caxis([0 1]);

hold on;

[qx,qy] = meshgrid(5:8:95,5:8:95);

[qu,qv] = sea_current(sea,qx,qy,0);

quiver(qx,qy,qu*25,qv*25,0,'b');

plot(astar_raw(:,1),astar_raw(:,2), ...
    'k--','LineWidth',1.5);

plot(path(:,1),path(:,2), ...
    'r','LineWidth',2.5);

plot(start(1),start(2), ...
    'go','MarkerSize',12,'LineWidth',3);

plot(target(1),target(2), ...
    'kx','MarkerSize',15,'LineWidth',3);

text(start(1)+3,start(2),'USV Start');

text(target(1)-22,target(2)-3,'Incident Location');

axis equal;
axis([1 map_size 1 map_size]);
grid on;

xlabel('X Position (m)');
ylabel('Y Position (m)');

title('Current-aware A*: raw path (dashed) and smoothed path (red)');

legend('Current field','Raw A*','Smoothed path', ...
       'Start','Incident', ...
       'Location','northwest');

%% =========================================================
% 12. FIGURE 2 - CLEARANCE PENALTY
% ==========================================================

figure('Name','Stage 2 - Clearance cost');

imagesc(penalty);

set(gca,'YDir','normal');

hold on;

plot(path(:,1),path(:,2), ...
    'w','LineWidth',2);

colorbar;

axis equal;
axis([1 map_size 1 map_size]);

xlabel('X Position (m)');
ylabel('Y Position (m)');

title('Soft clearance penalty used by A* (path in white)');

disp(' ');
disp('==============================================');
disp(' STAGE 2 ENHANCED + KNN COMPLETED');
disp('==============================================');

%% =========================================================
% LOCAL FUNCTION - LINE OF SIGHT
% ==========================================================

function ok = los_free(safe_map,p,q)

    n = ceil(norm(q-p)/0.5)+1;

    xs = linspace(p(1),q(1),n);
    ys = linspace(p(2),q(2),n);

    idx = sub2ind( ...
        size(safe_map), ...
        round(ys), ...
        round(xs));

    ok = all(safe_map(idx) == 0);

end