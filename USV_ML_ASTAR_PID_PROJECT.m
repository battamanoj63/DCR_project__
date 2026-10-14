%% AUTONOMOUS USV NAVIGATION PROJECT
% A* + CUSTOM KNN ML + PID + WIND + WATER CURRENT
% No external dataset.
% No Statistics and Machine Learning Toolbox.
% Compatible with MATLAB R2025b.

clc;
clear;
close all;

fprintf('\n');
fprintf('====================================================\n');
fprintf(' AUTONOMOUS USV NAVIGATION PROJECT\n');
fprintf(' A* + CUSTOM KNN ML + PID\n');
fprintf(' NO TOOLBOX REQUIRED\n');
fprintf('====================================================\n\n');

%% 1. ENVIRONMENT
mapSize = 100;
start = [5 5];
goal = [90 90];

map = zeros(mapSize,mapSize);

% Static obstacles
map(30:45,20:35) = 1;
map(55:70,45:60) = 1;
map(20:30,65:80) = 1;
map(70:85,15:28) = 1;

%% 2. SAFETY MAP
fprintf('Creating safety map...\n');
safetyMargin = 3;
safeMap = expandObstacles(map,safetyMargin);
safeMap(start(2),start(1)) = 0;
safeMap(goal(2),goal(1)) = 0;
fprintf('Safety map created.\n\n');

%% 3. CLEARANCE MAP
fprintf('Calculating obstacle clearance...\n');
clearanceMap = calculateClearance(map);
fprintf('Clearance map calculated.\n\n');

%% 4. A* PATH
fprintf('Calculating A* path...\n');
path = AStarPlanner(safeMap,clearanceMap,start,goal);

if isempty(path)
    error('A* could not find a safe path.');
end

astarPathLength = calculatePathLength(path);
fprintf('A* path calculated successfully.\n');
fprintf('A* path length = %.2f m\n\n',astarPathLength);

%% 5. PATH SIMPLIFICATION
smoothPath = simplifyPath(path,safeMap);
smoothPathLength = calculatePathLength(smoothPath);
fprintf('Smoothed path length = %.2f m\n\n',smoothPathLength);

%% 6. A* FIGURE
figure('Name','A* Safe Path','Color','w');
imagesc(map);
axis equal;
axis([1 mapSize 1 mapSize]);
set(gca,'YDir','normal');
hold on;
plot(path(:,1),path(:,2),'w-','LineWidth',2);
plot(smoothPath(:,1),smoothPath(:,2),'c-','LineWidth',3);
plot(start(1),start(2),'go','MarkerSize',10,'LineWidth',2);
plot(goal(1),goal(2),'rx','MarkerSize',14,'LineWidth',3);
text(start(1)+3,start(2),'USV START','FontWeight','bold');
text(goal(1)-20,goal(2)-3,'EMERGENCY ASSISTANCE','FontWeight','bold');
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('Safe A* Path Planning');
grid on;
legend('A* Path','Smoothed Path','USV Start','Emergency Assistance','Location','best');
hold off;

%% 7. MATHEMATICAL DYNAMIC OBSTACLES
numDynamicObstacles = 3;
dynamicStart = [48 18; 68 85; 84 42];
dynamicVelocity = [0.00 0.08; -0.06 0.00; 0.00 0.07];

%% 8. GENERATE SYNTHETIC ML DATA
fprintf('Generating ML training data...\n');

numSamples = 3000;

Distance = 2 + 40*rand(numSamples,1);
ClosingSpeed = -1 + 4*rand(numSamples,1);
RelativeAngle = -pi + 2*pi*rand(numSamples,1);
WindSpeed = 0.05 + 0.5*rand(numSamples,1);
CurrentSpeed = 0.05 + 0.6*rand(numSamples,1);
HeadingError = -pi + 2*pi*rand(numSamples,1);

TTC = 60*ones(numSamples,1);
positiveClosing = ClosingSpeed > 0;
TTC(positiveClosing) = Distance(positiveClosing)./ClosingSpeed(positiveClosing);
TTC = min(TTC,60);

Risk = double((Distance < 7) | ((TTC < 5) & (ClosingSpeed > 0.3)) | ((Distance < 11) & (abs(RelativeAngle) < pi/180*25) & (ClosingSpeed > 0.8)));

X = [Distance ClosingSpeed RelativeAngle WindSpeed CurrentSpeed HeadingError TTC];

featureMean = mean(X,1);
featureStd = std(X,0,1);
featureStd(featureStd == 0) = 1;
Xn = (X-featureMean)./featureStd;

rng(10);
order = randperm(numSamples);
numTrain = round(0.75*numSamples);
trainIndex = order(1:numTrain);
testIndex = order(numTrain+1:end);

Xtrain = Xn(trainIndex,:);
Ytrain = Risk(trainIndex);
Xtest = Xn(testIndex,:);
Ytest = Risk(testIndex);

K = 9;
testPrediction = zeros(length(testIndex),1);

for i = 1:length(testIndex)
    testPrediction(i) = knnPredict(Xtest(i,:),Xtrain,Ytrain,K);
end

testAccuracy = 100*mean(testPrediction == Ytest);

fprintf('ML Model: Custom K-Nearest Neighbors (KNN)\n');
fprintf('Training samples: %d\n',numTrain);
fprintf('Testing samples : %d\n',length(testIndex));
fprintf('ML Test Accuracy: %.2f %%\n',testAccuracy);
fprintf('ML training completed successfully.\n\n');

%% 9. PID AND USV PARAMETERS
Kp = 2.0;
Ki = 0.015;
Kd = 0.30;
maxYawRate = 0.70;
integralLimit = 2.5;

normalSpeed = 1.5;
riskSpeed = 1.15;
dangerSpeed = 0.75;

normalLookahead = 5;
riskLookahead = 8;

goalRadius = 2.0;

dt = 0.1;
simulationTime = 150;
numSteps = floor(simulationTime/dt)+1;

%% 10. RESULT ARRAYS
time = zeros(numSteps,1);
x = zeros(numSteps,1);
y = zeros(numSteps,1);
psi = zeros(numSteps,1);
speed = zeros(numSteps,1);
yawRate = zeros(numSteps,1);
headingError = zeros(numSteps,1);
trackingError = zeros(numSteps,1);
windSpeed = zeros(numSteps,1);
currentSpeed = zeros(numSteps,1);
riskProbability = zeros(numSteps,1);
riskState = zeros(numSteps,1);
nearestObstacleDistance = zeros(numSteps,1);

%% 11. INITIAL STATE
x(1) = start(1);
y(1) = start(2);

if size(smoothPath,1) >= 2
    firstTarget = smoothPath(2,:);
else
    firstTarget = goal;
end

psi(1) = atan2(firstTarget(2)-y(1),firstTarget(1)-x(1));

integralError = 0;
previousHeadingError = 0;
goalReached = false;
lastStep = numSteps;

%% 12. LIVE FIGURE
figure('Name','Autonomous USV Live Navigation','Color','w');
imagesc(map);
axis equal;
axis([1 mapSize 1 mapSize]);
set(gca,'YDir','normal');
hold on;

plot(smoothPath(:,1),smoothPath(:,2),'w-','LineWidth',3);
hActual = plot(x(1),y(1),'c-','LineWidth',2);
plot(start(1),start(2),'go','MarkerSize',10,'LineWidth',2);
plot(goal(1),goal(2),'rx','MarkerSize',14,'LineWidth',3);

hDynamic = gobjects(numDynamicObstacles,1);
for j = 1:numDynamicObstacles
    hDynamic(j) = plot(dynamicStart(j,1),dynamicStart(j,2),'ro','MarkerSize',10,'LineWidth',2);
end

hUSV = plot(x(1),y(1),'bo','MarkerSize',10,'LineWidth',2);

thetaCircle = linspace(0,2*pi,100);
safetyRadius = 7;
hSafety = plot(x(1)+safetyRadius*cos(thetaCircle),y(1)+safetyRadius*sin(thetaCircle),'b--','LineWidth',1.2);

xlabel('X Position (m)');
ylabel('Y Position (m)');
title('USV Navigation: A* + KNN ML + PID');
grid on;

%% 13. MAIN SIMULATION
fprintf('Starting USV simulation...\n');

for k = 1:numSteps

    time(k) = (k-1)*dt;

    % Mathematical wind
    windSpeed(k) = 0.06 + 0.02*sin(0.03*time(k));
    windDirection = pi/180*(25 + 8*sin(0.02*time(k)));
    windX = windSpeed(k)*cos(windDirection);
    windY = windSpeed(k)*sin(windDirection);

    % Mathematical water current
    currentSpeed(k) = 0.08 + 0.025*sin(0.025*time(k)+1);
    currentDirection = pi/180*(145 + 7*cos(0.018*time(k)));
    currentX = currentSpeed(k)*cos(currentDirection);
    currentY = currentSpeed(k)*sin(currentDirection);

    % Closest path point
    pathDistance = sqrt((smoothPath(:,1)-x(k)).^2 + (smoothPath(:,2)-y(k)).^2);
    [~,closestIndex] = min(pathDistance);
    trackingError(k) = pathDistance(closestIndex);

    % Dynamic obstacle positions
    obstaclePositions = zeros(numDynamicObstacles,2);
    for j = 1:numDynamicObstacles
        obstaclePositions(j,:) = dynamicObstaclePosition(dynamicStart(j,:),dynamicVelocity(j,:),time(k));
    end

    % Nearest dynamic obstacle
    dynamicDistances = sqrt((obstaclePositions(:,1)-x(k)).^2 + (obstaclePositions(:,2)-y(k)).^2);
    [nearestDistance,nearestIndex] = min(dynamicDistances);
    nearestObstacleDistance(k) = nearestDistance;
    nearestObstacle = obstaclePositions(nearestIndex,:);

    % Closing speed
    if k == 1
        closingSpeed = 0;
    else
        closingSpeed = (nearestObstacleDistance(k-1)-nearestDistance)/dt;
    end

    % Relative angle
    relativeBearing = atan2(nearestObstacle(2)-y(k),nearestObstacle(1)-x(k));
    relativeAngle = wrapToPiLocal(relativeBearing-psi(k));

    % Time to collision
    if closingSpeed > 0
        ttc = nearestDistance/closingSpeed;
    else
        ttc = 60;
    end
    ttc = min(max(ttc,0),60);

    % Temporary heading
    tempIndex = min(closestIndex+normalLookahead,size(smoothPath,1));
    tempHeading = atan2(smoothPath(tempIndex,2)-y(k),smoothPath(tempIndex,1)-x(k));
    tempHeadingError = wrapToPiLocal(tempHeading-psi(k));

    % ML feature vector
    currentFeatures = [nearestDistance closingSpeed relativeAngle windSpeed(k) currentSpeed(k) tempHeadingError ttc];
    currentFeaturesNormalized = (currentFeatures-featureMean)./featureStd;

    [predictedRisk,probability] = knnPredictProbability(currentFeaturesNormalized,Xtrain,Ytrain,K);
    riskProbability(k) = probability;
    riskState(k) = predictedRisk;

    % Adaptive speed
    if nearestDistance < 5
        speed(k) = dangerSpeed;
        lookahead = riskLookahead;
        riskState(k) = 1;
    elseif predictedRisk == 1
        speed(k) = riskSpeed;
        lookahead = riskLookahead;
    else
        speed(k) = normalSpeed;
        lookahead = normalLookahead;
    end

    % Target waypoint
    targetIndex = min(closestIndex+lookahead,size(smoothPath,1));
    targetPoint = smoothPath(targetIndex,:);
    desiredHeading = atan2(targetPoint(2)-y(k),targetPoint(1)-x(k));

    % Dynamic obstacle repulsion
    avoidanceVector = [0 0];

    for j = 1:numDynamicObstacles
        dx = x(k)-obstaclePositions(j,1);
        dy = y(k)-obstaclePositions(j,2);
        d = sqrt(dx^2+dy^2);

        if d < 12 && d > 0.01
            strength = ((12-d)/12)^2;
            avoidanceVector = avoidanceVector + strength*[dx dy]/d;
        end
    end

    % Static obstacle repulsion
    [staticVector,staticDistance] = staticObstacleRepulsion([x(k) y(k)],map);

    if staticDistance < 9
        avoidanceVector = avoidanceVector + 1.2*staticVector;
        speed(k) = min(speed(k),0.9);
    end

    % Combine path following and avoidance
    desiredVector = [cos(desiredHeading) sin(desiredHeading)] + avoidanceVector;

    if norm(desiredVector) > 0.001
        desiredHeading = atan2(desiredVector(2),desiredVector(1));
    end

    % PID heading controller
    headingError(k) = wrapToPiLocal(desiredHeading-psi(k));
    integralError = integralError + headingError(k)*dt;
    integralError = max(-integralLimit,min(integralError,integralLimit));

    if k == 1
        derivativeError = 0;
    else
        derivativeError = (headingError(k)-previousHeadingError)/dt;
    end

    yawCommand = Kp*headingError(k) + Ki*integralError + Kd*derivativeError;
    yawRate(k) = max(-maxYawRate,min(yawCommand,maxYawRate));

    % USV motion model
    if k < numSteps
        psi(k+1) = wrapToPiLocal(psi(k)+yawRate(k)*dt);

        vehicleX = speed(k)*cos(psi(k));
        vehicleY = speed(k)*sin(psi(k));

        x(k+1) = x(k) + (vehicleX+windX+currentX)*dt;
        y(k+1) = y(k) + (vehicleY+windY+currentY)*dt;

        x(k+1) = max(1,min(mapSize,x(k+1)));
        y(k+1) = max(1,min(mapSize,y(k+1)));
    end

    % Goal check
    goalDistance = sqrt((goal(1)-x(k))^2 + (goal(2)-y(k))^2);

    if goalDistance <= goalRadius
        goalReached = true;
        lastStep = k;
        fprintf('USV reached the emergency assistance location.\n');
        break;
    end

    previousHeadingError = headingError(k);

    % Animation
    if mod(k,5) == 0
        set(hActual,'XData',x(1:k),'YData',y(1:k));
        set(hUSV,'XData',x(k),'YData',y(k));

        for j = 1:numDynamicObstacles
            set(hDynamic(j),'XData',obstaclePositions(j,1),'YData',obstaclePositions(j,2));
        end

        set(hSafety,'XData',x(k)+safetyRadius*cos(thetaCircle),'YData',y(k)+safetyRadius*sin(thetaCircle));

        if riskState(k) == 1
            title(sprintf('USV NAVIGATION | ML RISK | Speed %.2f m/s',speed(k)));
        else
            title(sprintf('USV NAVIGATION | ML SAFE | Speed %.2f m/s',speed(k)));
        end

        drawnow;
    end
end

%% 14. TRIM RESULTS
x = x(1:lastStep);
y = y(1:lastStep);
psi = psi(1:lastStep);
time = time(1:lastStep);
speed = speed(1:lastStep);
yawRate = yawRate(1:lastStep);
headingError = headingError(1:lastStep);
trackingError = trackingError(1:lastStep);
windSpeed = windSpeed(1:lastStep);
currentSpeed = currentSpeed(1:lastStep);
riskProbability = riskProbability(1:lastStep);
riskState = riskState(1:lastStep);
nearestObstacleDistance = nearestObstacleDistance(1:lastStep);

%% 15. PERFORMANCE METRICS
RMSE = sqrt(mean(trackingError.^2));
MAE = mean(abs(trackingError));
maxTrackingError = max(trackingError);
finalPosition = [x(end) y(end)];
finalGoalError = norm(finalPosition-goal);
actualPathLength = calculatePathLength([x y]);
finalHeadingError = abs(headingError(end));
riskPercentage = 100*mean(riskState == 1);

fprintf('\n');
fprintf('====================================================\n');
fprintf(' FINAL PROJECT RESULTS\n');
fprintf('====================================================\n');
fprintf('ML Test Accuracy     : %.2f %%\n',testAccuracy);
fprintf('A* Path Length       : %.2f m\n',astarPathLength);
fprintf('Smoothed Path Length : %.2f m\n',smoothPathLength);
fprintf('Actual Path Length   : %.2f m\n',actualPathLength);
fprintf('Tracking RMSE        : %.3f m\n',RMSE);
fprintf('Tracking MAE         : %.3f m\n',MAE);
fprintf('Maximum Tracking Err : %.3f m\n',maxTrackingError);
fprintf('Final Position       : (%.2f, %.2f)\n',finalPosition(1),finalPosition(2));
fprintf('Target Position      : (%.2f, %.2f)\n',goal(1),goal(2));
fprintf('Final Goal Error     : %.3f m\n',finalGoalError);
fprintf('Final Heading Error  : %.3f rad\n',finalHeadingError);
fprintf('Simulation Time      : %.2f s\n',time(end));
fprintf('ML Risk Time         : %.2f %%\n',riskPercentage);

if goalReached
    fprintf('Goal Status          : REACHED\n');
else
    fprintf('Goal Status          : NOT REACHED\n');
end

fprintf('====================================================\n');

%% 16. FINAL TRAJECTORY
figure('Name','Final USV Navigation Result','Color','w');
imagesc(map);
axis equal;
axis([1 mapSize 1 mapSize]);
set(gca,'YDir','normal');
hold on;
plot(smoothPath(:,1),smoothPath(:,2),'w-','LineWidth',3);
plot(x,y,'c-','LineWidth',3);
plot(start(1),start(2),'go','MarkerSize',12,'LineWidth',3);
plot(goal(1),goal(2),'rx','MarkerSize',15,'LineWidth',3);
xlabel('X Position (m)');
ylabel('Y Position (m)');
title('FINAL USV NAVIGATION: A* + KNN ML + PID');
legend('A* Planned Path','Actual USV Trajectory','USV Start','Emergency Assistance','Location','best');
grid on;
hold off;

%% 17. TRACKING ERROR
figure('Name','Tracking Error','Color','w');
plot(time,trackingError,'LineWidth',2);
xlabel('Time (s)');
ylabel('Tracking Error (m)');
title('USV Path Tracking Error');
grid on;

%% 18. HEADING ERROR
figure('Name','Heading Error','Color','w');
plot(time,headingError*180/pi,'LineWidth',2);
xlabel('Time (s)');
ylabel('Heading Error (degrees)');
title('PID Heading Error');
grid on;

%% 19. WIND AND CURRENT
figure('Name','Environmental Disturbances','Color','w');
plot(time,windSpeed,'LineWidth',2);
hold on;
plot(time,currentSpeed,'LineWidth',2);
xlabel('Time (s)');
ylabel('Speed (m/s)');
title('Mathematical Wind and Water Current');
legend('Wind','Water Current','Location','best');
grid on;
hold off;

%% 20. ML RISK
figure('Name','ML Collision Risk','Color','w');
plot(time,riskProbability,'LineWidth',2);
hold on;
yline(0.5,'--','Risk Threshold');
xlabel('Time (s)');
ylabel('Risk Probability');
title('Custom KNN Collision Risk Prediction');
ylim([0 1]);
grid on;
hold off;

%% 21. SAVE
save('USV_ML_ASTAR_PID_RESULTS.mat','path','smoothPath','x','y','psi','time','speed','yawRate','trackingError','headingError','windSpeed','currentSpeed','riskProbability','riskState','astarPathLength','smoothPathLength','actualPathLength','RMSE','MAE','maxTrackingError','finalPosition','finalGoalError','testAccuracy');

fprintf('\nResults saved as USV_ML_ASTAR_PID_RESULTS.mat\n');
fprintf('PROJECT COMPLETED.\n');

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function safeMap = expandObstacles(map,safetyMargin)
[rows,cols] = size(map);
safeMap = map;
for yy = 1:rows
    for xx = 1:cols
        if map(yy,xx) == 1
            xmin = max(1,xx-safetyMargin);
            xmax = min(cols,xx+safetyMargin);
            ymin = max(1,yy-safetyMargin);
            ymax = min(rows,yy+safetyMargin);
            safeMap(ymin:ymax,xmin:xmax) = 1;
        end
    end
end
end

function clearanceMap = calculateClearance(map)
[rows,cols] = size(map);
clearanceMap = inf(rows,cols);
[oy,ox] = find(map == 1);
if isempty(ox)
    return;
end
for yy = 1:rows
    for xx = 1:cols
        if map(yy,xx) == 1
            clearanceMap(yy,xx) = 0;
        else
            d = sqrt((ox-xx).^2 + (oy-yy).^2);
            clearanceMap(yy,xx) = min(d);
        end
    end
end
end

function path = AStarPlanner(map,clearanceMap,start,goal)
[rows,cols] = size(map);
openList = start;
closedList = false(rows,cols);
gCost = inf(rows,cols);
fCost = inf(rows,cols);
parentX = zeros(rows,cols);
parentY = zeros(rows,cols);

gCost(start(2),start(1)) = 0;
fCost(start(2),start(1)) = heuristic(start,goal);
found = false;

neighbors = [-1 -1; 0 -1; 1 -1; -1 0; 1 0; -1 1; 0 1; 1 1];

while ~isempty(openList)
    bestIndex = 1;
    bestCost = inf;

    for ii = 1:size(openList,1)
        cx = openList(ii,1);
        cy = openList(ii,2);
        if fCost(cy,cx) < bestCost
            bestCost = fCost(cy,cx);
            bestIndex = ii;
        end
    end

    current = openList(bestIndex,:);
    openList(bestIndex,:) = [];

    cx = current(1);
    cy = current(2);

    if isequal(current,goal)
        found = true;
        break;
    end

    closedList(cy,cx) = true;

    for nn = 1:size(neighbors,1)
        dx = neighbors(nn,1);
        dy = neighbors(nn,2);
        nx = cx + dx;
        ny = cy + dy;

        if nx < 1 || nx > cols || ny < 1 || ny > rows
            continue;
        end

        if map(ny,nx) == 1 || closedList(ny,nx)
            continue;
        end

        if dx ~= 0 && dy ~= 0
            if map(cy,nx) == 1 || map(ny,cx) == 1
                continue;
            end
        end

        moveCost = sqrt(dx^2 + dy^2);
        clearance = max(clearanceMap(ny,nx),1);
        penalty = 1 + 3/(clearance^2);
        tentativeG = gCost(cy,cx) + moveCost*penalty;

        if tentativeG < gCost(ny,nx)
            parentX(ny,nx) = cx;
            parentY(ny,nx) = cy;
            gCost(ny,nx) = tentativeG;
            fCost(ny,nx) = tentativeG + heuristic([nx ny],goal);

            if ~ismember([nx ny],openList,'rows')
                openList(end+1,:) = [nx ny];
            end
        end
    end
end

if ~found
    path = [];
    return;
end

path = goal;
current = goal;

while ~isequal(current,start)
    px = parentX(current(2),current(1));
    py = parentY(current(2),current(1));

    if px == 0 || py == 0
        path = [];
        return;
    end

    current = [px py];
    path = [current; path];
end
end

function h = heuristic(node,goal)
h = sqrt((node(1)-goal(1))^2 + (node(2)-goal(2))^2);
end

function L = calculatePathLength(path)
if size(path,1) < 2
    L = 0;
    return;
end
differences = diff(path,1,1);
L = sum(sqrt(sum(differences.^2,2)));
end

function simplified = simplifyPath(path,map)
if size(path,1) <= 2
    simplified = path;
    return;
end

simplified = path(1,:);
currentIndex = 1;

while currentIndex < size(path,1)
    bestIndex = currentIndex + 1;

    for jj = size(path,1):-1:(currentIndex+1)
        if lineIsFree(path(currentIndex,:),path(jj,:),map)
            bestIndex = jj;
            break;
        end
    end

    simplified(end+1,:) = path(bestIndex,:);
    currentIndex = bestIndex;
end
end

function free = lineIsFree(p1,p2,map)
distance = norm(p2-p1);
n = max(2,ceil(distance*3));
free = true;

for ii = 0:n
    alpha = ii/n;
    xx = round(p1(1) + alpha*(p2(1)-p1(1)));
    yy = round(p1(2) + alpha*(p2(2)-p1(2)));

    if xx < 1 || xx > size(map,2) || yy < 1 || yy > size(map,1)
        free = false;
        return;
    end

    if map(yy,xx) == 1
        free = false;
        return;
    end
end
end

function angle = wrapToPiLocal(angle)
angle = mod(angle + pi,2*pi) - pi;
end

function position = dynamicObstaclePosition(initialPosition,velocity,t)
lowerLimit = 5;
upperLimit = 95;
rangeValue = upperLimit-lowerLimit;
position = zeros(1,2);

for ii = 1:2
    raw = initialPosition(ii) + velocity(ii)*t;
    q = mod(raw-lowerLimit,2*rangeValue);

    if q <= rangeValue
        position(ii) = lowerLimit + q;
    else
        position(ii) = upperLimit - (q-rangeValue);
    end
end
end

function prediction = knnPredict(sample,Xtrain,Ytrain,K)
differences = Xtrain - sample;
distances = sqrt(sum(differences.^2,2));
[~,order] = sort(distances,'ascend');
K = min(K,length(order));
labels = Ytrain(order(1:K));
prediction = double(mean(labels) >= 0.5);
end

function [prediction,probability] = knnPredictProbability(sample,Xtrain,Ytrain,K)
differences = Xtrain - sample;
distances = sqrt(sum(differences.^2,2));
[~,order] = sort(distances,'ascend');
K = min(K,length(order));
labels = Ytrain(order(1:K));
probability = mean(labels);
prediction = double(probability >= 0.5);
end

function [repulsion,minDistance] = staticObstacleRepulsion(position,map)
repulsion = [0 0];
minDistance = inf;

[oy,ox] = find(map == 1);

if isempty(ox)
    return;
end

distances = sqrt((ox-position(1)).^2 + (oy-position(2)).^2);
[minDistance,index] = min(distances);

if minDistance < 10
    nearestPoint = [ox(index) oy(index)];
    difference = position-nearestPoint;
    d = norm(difference);

    if d > 0.001
        repulsion = difference/d;
    end
end
end
