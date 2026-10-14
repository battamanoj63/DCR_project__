clc;
clear;
close all;

%% =========================================================
% STAGE 4 (ENHANCED) - 3D DYNAMIC USV NAVIGATION
%
% A* PATH + PID GUIDANCE + MOVING OBSTACLES + REALISTIC SEA
% + CUSTOM KNN ML COLLISION-RISK PREDICTION
%
% ML:
%   - Custom KNN classifier
%   - No Statistics and Machine Learning Toolbox required
%   - Predicts collision risk from dynamic-obstacle conditions
%   - Adaptive speed
%   - Adaptive look-ahead
%   - Adaptive avoidance strength
%
% Existing Stage 4 simulation, figures and sea model retained.
% =========================================================

disp('==============================================');
disp(' STAGE 4 ENHANCED - 3D DYNAMIC SIMULATION');
disp(' + CUSTOM KNN ML COLLISION-RISK PREDICTION');
disp('==============================================');

%% =========================================================
% 0. USER OPTIONS
% =========================================================

preset             = 'moderate';   % 'calm' | 'moderate' | 'rough'
save_video         = false;        % true -> writes usv_stage4.mp4
update_water_every = 2;            % redraw the water every n steps
vis_z              = 2;            % vertical exaggeration of the waves

%% =========================================================
% 1. LOAD PATH, MAP AND ML MODEL
% =========================================================

if ~isfile('astar_path.mat')
    error('astar_path.mat not found. Run Stage 2 first.');
end

if ~isfile('knn_risk_model.mat')
    error(['knn_risk_model.mat not found. ' ...
           'Run the ML-enabled Stage 2 first.']);
end

S = load('astar_path.mat');
path = S.path;

M = load('knn_risk_model.mat');
knnModel = M.knnModel;

[~,~,rects] = make_map(4);

mapSize = 100;
start   = path(1,:);
target  = path(end,:);

%% =========================================================
% 2. SIMULATION AND SEA
% =========================================================

dt = 0.15;
T  = 250;
N  = round(T/dt)+1;
t  = (0:N-1)*dt;

sea = make_sea(preset,T,dt,10);
rng(10);

%% =========================================================
% 3. USV AND CONTROLLER PARAMETERS
% =========================================================

USV_speed       = 1.2;
maxYawRate      = 0.7;
Tr              = 0.5;
lookAhead       = 5;
targetThreshold = 3;

Kp = 2.8;
Ki = 0.01;
Kd = 0.40;

gps_sigma     = 0.3;
compass_sigma = deg2rad(1);
gps_every     = max(1,round(1/dt));

boatLength = 5;
boatWidth  = 2.5;
boatHeight = 1;

%% =========================================================
% 4. INITIAL STATE
% =========================================================

x = zeros(1,N);
y = zeros(1,N);
heading = zeros(1,N);

x(1) = start(1);
y(1) = start(2);

heading(1) = atan2( ...
    path(min(5,end),2)-start(2), ...
    path(min(5,end),1)-start(1));

r_act        = 0;
integralError = 0;
previousError = 0;
d_filt        = 0;

pathIdx = 1;
meas_pos = start;

roll_s  = 0;
pitch_s = 0;

%% =========================================================
% 5. DYNAMIC OBSTACLES
% =========================================================

numObstacles = 5;

obstacleX    = [25 45 65 30 75];
obstacleY    = [25 55 35 75 70];
obstacleSize = [7 6 8 7 6];

obstacleVx = zeros(1,numObstacles);
obstacleVy = zeros(1,numObstacles);

for i = 1:numObstacles
    ang = 2*pi*rand;
    spd = 0.3 + 0.4*rand;

    obstacleVx(i) = spd*cos(ang);
    obstacleVy(i) = spd*sin(ang);
end

%% =========================================================
% 6. DATA LOGS
% =========================================================

windLog    = zeros(1,N);
currentLog = zeros(1,N);

trackingError = zeros(1,N);
headingErrLog = zeros(1,N);
yawCommand    = zeros(1,N);

heaveLog = zeros(1,N);
rollLog  = zeros(1,N);
pitchLog = zeros(1,N);

minSepLog = nan(1,N);

% ---------------------------------------------------------
% ML LOGS
% ---------------------------------------------------------

riskProbabilityLog = zeros(1,N);
riskLabelLog       = zeros(1,N);

adaptiveSpeedLog       = zeros(1,N);
adaptiveLookAheadLog   = zeros(1,N);
avoidanceStrengthLog   = zeros(1,N);

mlDistanceLog      = zeros(1,N);
mlClosingSpeedLog  = zeros(1,N);
mlTTClog           = zeros(1,N);

collisions = 0;

last    = N;
reached = false;

%% =========================================================
% 7. 3D FIGURE
% =========================================================

fig = figure( ...
    'Color','k', ...
    'Name','Autonomous USV - Dynamic Navigation');

ax = axes(fig);
hold(ax,'on');

set(ax, ...
    'Color',[0.01 0.04 0.10], ...
    'XColor','w', ...
    'YColor','w', ...
    'ZColor','w');

axis(ax,[0 mapSize 0 mapSize -3 8]);
pbaspect(ax,[1 1 0.3]);
grid(ax,'on');

xlabel('X Position (m)','Color','w');
ylabel('Y Position (m)','Color','w');
zlabel('Height (m)','Color','w');

view(ax,45,50);

title( ...
    sprintf('3D Dynamic Navigation - Autonomous USV (%s sea)',sea.name), ...
    'Color','w', ...
    'FontSize',14, ...
    'FontWeight','bold');

%% ---- water surface

[Xw,Yw] = meshgrid(0:2:mapSize,0:2:mapSize);

Zw = vis_z*sea_waves(sea,Xw,Yw,0);

waterSurface = surf( ...
    ax,Xw,Yw,Zw, ...
    'FaceAlpha',0.75, ...
    'EdgeColor','none');

colormap(ax,[ ...
    linspace(0.00,0.25,64)' ...
    linspace(0.10,0.55,64)' ...
    linspace(0.25,0.80,64)' ]);

caxis(ax,vis_z*[-1.2 1.2]);

%% ---- static obstacles from the map

for i = 1:size(rects,1)

    patch( ...
        ax, ...
        'Vertices',box_vertices( ...
            mean(rects(i,1:2)), ...
            mean(rects(i,3:4)), ...
            rects(i,2)-rects(i,1)+1, ...
            rects(i,4)-rects(i,3)+1, ...
            2.5, ...
            0), ...
        'Faces',box_faces(), ...
        'FaceColor',[0.45 0.45 0.5], ...
        'EdgeColor',[0.8 0.8 0.8], ...
        'LineWidth',1);
end

%% ---- planned path, start, target

plot3( ...
    ax, ...
    path(:,1), ...
    path(:,2), ...
    0.6*ones(size(path,1),1), ...
    '--', ...
    'LineWidth',1.5, ...
    'Color',[0.7 0.7 0.7]);

plot3( ...
    ax, ...
    start(1), ...
    start(2), ...
    1, ...
    'o', ...
    'MarkerSize',12, ...
    'LineWidth',3, ...
    'Color',[1 0.5 0]);

text( ...
    ax, ...
    start(1), ...
    start(2), ...
    3.5, ...
    'USV START', ...
    'Color','w', ...
    'FontWeight','bold');

plot3( ...
    ax, ...
    target(1), ...
    target(2), ...
    1, ...
    'x', ...
    'MarkerSize',15, ...
    'LineWidth',4, ...
    'Color','y');

text( ...
    ax, ...
    target(1)-12, ...
    target(2), ...
    3.5, ...
    'EMERGENCY ASSISTANCE', ...
    'Color','y', ...
    'FontWeight','bold');

%% ---- USV

V0 = hull_local(boatLength,boatWidth,boatHeight);

boat = patch( ...
    ax, ...
    'Vertices',V0, ...
    'Faces',box_faces(), ...
    'FaceColor',[0.85 0.85 0.9], ...
    'EdgeColor','w', ...
    'LineWidth',1.5);

%% ---- moving obstacles

obstacleGraphics = gobjects(1,numObstacles);

for i = 1:numObstacles

    obstacleGraphics(i) = patch( ...
        ax, ...
        'Vertices',box_vertices( ...
            obstacleX(i), ...
            obstacleY(i), ...
            obstacleSize(i), ...
            0.65*obstacleSize(i), ...
            2, ...
            0), ...
        'Faces',box_faces(), ...
        'FaceColor',[0.8 0.05 0.03], ...
        'EdgeColor',[1 0.3 0.2], ...
        'LineWidth',1.2);
end

%% ---- trajectory

trajectory = plot3( ...
    ax, ...
    x(1), ...
    y(1), ...
    1, ...
    'Color',[0 0.8 1], ...
    'LineWidth',2.5);

%% ---- wind arrows

windArrows = gobjects(1,4);

wpos = [15 15; 15 85; 85 15; 85 85];

for i = 1:4

    windArrows(i) = quiver3( ...
        ax, ...
        wpos(i,1), ...
        wpos(i,2), ...
        5, ...
        0,0,0, ...
        'Color',[1 0 1], ...
        'LineWidth',1.5, ...
        'MaxHeadSize',2);
end

%% ---- current arrows

[cgx,cgy] = meshgrid([20 50 80],[20 50 80]);

currentArrows = gobjects(1,numel(cgx));

for i = 1:numel(cgx)

    currentArrows(i) = quiver3( ...
        ax, ...
        cgx(i), ...
        cgy(i), ...
        1.5, ...
        0,0,0, ...
        'Color',[0.2 1 0.3], ...
        'LineWidth',1.5, ...
        'MaxHeadSize',2);
end

%% ---- status panel

statusText = text( ...
    ax, ...
    3,94,8, ...
    '', ...
    'Color','w', ...
    'FontSize',10, ...
    'FontWeight','bold', ...
    'BackgroundColor',[0 0 0], ...
    'Margin',6);

%% ---- optional video

if save_video

    vw = VideoWriter('usv_stage4.mp4','MPEG-4');
    vw.FrameRate = 20;
    open(vw);

end

%% =========================================================
% 8. SIMULATION
% =========================================================

for k = 1:N-1

    pos = [x(k) y(k)];

    %% -----------------------------------------------------
    % TARGET REACHED
    % ------------------------------------------------------

    distanceTarget = norm(target-pos);

    if distanceTarget < targetThreshold

        reached = true;
        last = k;

        x(k+1:end) = x(k);
        y(k+1:end) = y(k);
        heading(k+1:end) = heading(k);

        break;
    end

    %% -----------------------------------------------------
    % SENSORS
    % ------------------------------------------------------

    if mod(k-1,gps_every) == 0

        meas_pos = pos + gps_sigma*randn(1,2);

    end

    heading_meas = heading(k) + compass_sigma*randn;

    %% -----------------------------------------------------
    % GUIDANCE: LOOK-AHEAD POINT
    % ------------------------------------------------------

    hi = min(size(path,1),pathIdx+15);

    dd = hypot( ...
        path(pathIdx:hi,1)-meas_pos(1), ...
        path(pathIdx:hi,2)-meas_pos(2));

    [~,m] = min(dd);

    pathIdx = pathIdx+m-1;

    % The lookAhead value will be adapted by ML below.
    % Start with the original value.

    currentLookAhead = lookAhead;

    li = pathIdx;
    acc = 0;

    while li < size(path,1) && acc < currentLookAhead

        acc = acc + norm(path(li+1,:)-path(li,:));
        li = li+1;

    end

    waypoint = path(li,:);

    desiredHeading = atan2( ...
        waypoint(2)-meas_pos(2), ...
        waypoint(1)-meas_pos(1));

    %% -----------------------------------------------------
    % ENVIRONMENT
    % ------------------------------------------------------

    tk = t(k);

    [wx,wy,windSpeed] = sea_wind(sea,tk);

    [cx,cy] = sea_current( ...
        sea, ...
        x(k), ...
        y(k), ...
        tk);

    [eta,dEx,dEy] = sea_waves( ...
        sea, ...
        x(k), ...
        y(k), ...
        tk);

    windLog(k)    = windSpeed;
    currentLog(k) = hypot(cx,cy);

    long_slope = ...
        dEx*cos(heading(k)) + ...
        dEy*sin(heading(k));

    lat_slope = ...
        -dEx*sin(heading(k)) + ...
        dEy*cos(heading(k));

    % boat attitude follows the wave slope with some hull inertia

    pitch_s = pitch_s + ...
        dt*(atan(long_slope)-pitch_s)/0.6;

    roll_s = roll_s + ...
        dt*(atan(lat_slope)-roll_s)/0.6;

    heaveLog(k) = eta;
    rollLog(k)  = roll_s;
    pitchLog(k) = pitch_s;

    %% -----------------------------------------------------
    % MOVING OBSTACLES
    % ------------------------------------------------------

    for i = 1:numObstacles

        ang = atan2( ...
            obstacleVy(i), ...
            obstacleVx(i)) + ...
            0.15*sqrt(dt)*randn;

        spd = hypot( ...
            obstacleVx(i), ...
            obstacleVy(i));

        obstacleVx(i) = spd*cos(ang);
        obstacleVy(i) = spd*sin(ang);

        obstacleX(i) = ...
            obstacleX(i) + obstacleVx(i)*dt;

        obstacleY(i) = ...
            obstacleY(i) + obstacleVy(i)*dt;

        if obstacleX(i) < 8 || obstacleX(i) > 92
            obstacleVx(i) = -obstacleVx(i);
        end

        if obstacleY(i) < 8 || obstacleY(i) > 92
            obstacleVy(i) = -obstacleVy(i);
        end

        obstacleX(i) = ...
            max(8,min(92,obstacleX(i)));

        obstacleY(i) = ...
            max(8,min(92,obstacleY(i)));

    end

    %% =====================================================
    % ML COLLISION-RISK PREDICTION
    % ======================================================

    % Find the closest moving obstacle.

    nearestObstacle = 1;
    nearestDistance = inf;

    for i = 1:numObstacles

        d = hypot( ...
            obstacleX(i)-x(k), ...
            obstacleY(i)-y(k));

        if d < nearestDistance

            nearestDistance = d;
            nearestObstacle = i;

        end
    end

    % Position of closest obstacle.

    obstacleDX = ...
        obstacleX(nearestObstacle)-x(k);

    obstacleDY = ...
        obstacleY(nearestObstacle)-y(k);

    mlDistance = hypot(obstacleDX,obstacleDY);

    % Unit vector from USV to obstacle.

    if mlDistance > 0.1

        unitX = obstacleDX/mlDistance;
        unitY = obstacleDY/mlDistance;

    else

        unitX = 0;
        unitY = 0;

    end

    % Nominal USV velocity.

    usvVx = USV_speed*cos(heading(k));
    usvVy = USV_speed*sin(heading(k));

    % Relative obstacle velocity with respect to USV.

    relativeVx = ...
        obstacleVx(nearestObstacle)-usvVx;

    relativeVy = ...
        obstacleVy(nearestObstacle)-usvVy;

    % Positive value means obstacle is approaching.

    closingSpeed = ...
        -(relativeVx*unitX + relativeVy*unitY);

    closingSpeed = max(closingSpeed,0);

    % Time to collision.

    if closingSpeed > 0.05

        TTC = mlDistance/closingSpeed;

    else

        TTC = 60;

    end

    TTC = min(TTC,60);

    % Relative angle between USV heading and obstacle direction.

    relativeAngle = rad2deg( ...
        wrap_pi( ...
        atan2(obstacleDY,obstacleDX)-heading_meas));

    % Heading error before avoidance.

    baseHeadingError = rad2deg( ...
        wrap_pi(desiredHeading-heading_meas));

    % Wind/current speeds.

    windFeature    = windSpeed;
    currentFeature = hypot(cx,cy);

    % -----------------------------------------------------
    % KNN FEATURE VECTOR
    % -----------------------------------------------------

    mlFeatures = [ ...
        mlDistance, ...
        closingSpeed, ...
        relativeAngle, ...
        windFeature, ...
        currentFeature, ...
        baseHeadingError, ...
        TTC];

    % -----------------------------------------------------
    % CUSTOM KNN PREDICTION
    % -----------------------------------------------------

    [riskLabel,riskProbability] = ...
        knn_predict(knnModel,mlFeatures);

    % -----------------------------------------------------
    % ADAPTIVE NAVIGATION
    % -----------------------------------------------------

    if riskProbability >= 0.75

        % HIGH COLLISION RISK

        desiredSpeed = 0.55;
        currentLookAhead = 10;
        avoidanceStrength = 5.0;

    elseif riskProbability >= 0.50

        % MEDIUM COLLISION RISK

        desiredSpeed = 0.75;
        currentLookAhead = 8;
        avoidanceStrength = 3.5;

    else

        % SAFE

        desiredSpeed = USV_speed;
        currentLookAhead = lookAhead;
        avoidanceStrength = 2.5;

    end

    % Save ML logs.

    riskProbabilityLog(k) = riskProbability;
    riskLabelLog(k)       = riskLabel;

    adaptiveSpeedLog(k)     = desiredSpeed;
    adaptiveLookAheadLog(k) = currentLookAhead;
    avoidanceStrengthLog(k) = avoidanceStrength;

    mlDistanceLog(k)     = mlDistance;
    mlClosingSpeedLog(k) = closingSpeed;
    mlTTClog(k)          = TTC;

    %% -----------------------------------------------------
    % RECALCULATE WAYPOINT WITH ML LOOK-AHEAD
    % ------------------------------------------------------

    li = pathIdx;
    acc = 0;

    while li < size(path,1) && acc < currentLookAhead

        acc = acc + norm(path(li+1,:)-path(li,:));
        li = li+1;

    end

    waypoint = path(li,:);

    desiredHeading = atan2( ...
        waypoint(2)-meas_pos(2), ...
        waypoint(1)-meas_pos(1));

    %% -----------------------------------------------------
    % OBSTACLE AVOIDANCE
    % ------------------------------------------------------

    avoidanceX = 0;
    avoidanceY = 0;

    minSep = inf;

    for i = 1:numObstacles

        diffv = ...
            pos-[obstacleX(i) obstacleY(i)];

        distance = max(norm(diffv),0.1);

        safetyDistance = ...
            obstacleSize(i)+7;

        minSep = min( ...
            minSep, ...
            distance-obstacleSize(i)/2);

        if distance < safetyDistance

            repulsion = ...
                (safetyDistance-distance)/safetyDistance;

            avoidanceX = ...
                avoidanceX + ...
                repulsion*diffv(1)/distance;

            avoidanceY = ...
                avoidanceY + ...
                repulsion*diffv(2)/distance;

        end

        if distance < obstacleSize(i)/2+1.5

            collisions = collisions+1;

        end

    end

    % Static obstacles.

    [sx,sy] = rect_repulsion(pos,rects,6);

    avoidanceX = avoidanceX+sx;
    avoidanceY = avoidanceY+sy;

    minSepLog(k) = minSep;

    % -----------------------------------------------------
    % ML ADAPTIVE AVOIDANCE
    % ------------------------------------------------------

    desiredHeading = atan2( ...
        sin(desiredHeading) + ...
        avoidanceStrength*avoidanceY, ...
        cos(desiredHeading) + ...
        avoidanceStrength*avoidanceX);

    %% -----------------------------------------------------
    % PID HEADING CONTROL
    % ------------------------------------------------------

    headErr = ...
        wrap_pi(desiredHeading-heading_meas);

    headingErrLog(k) = headErr;

    integralError = max( ...
        -4, ...
        min(4,integralError+headErr*dt));

    d_raw = ...
        (headErr-previousError)/dt;

    d_filt = ...
        0.8*d_filt + 0.2*d_raw;

    previousError = headErr;

    yaw = ...
        Kp*headErr + ...
        Ki*integralError + ...
        Kd*d_filt;

    yaw = max( ...
        -maxYawRate, ...
        min(maxYawRate,yaw));

    yawCommand(k) = yaw;

    %% -----------------------------------------------------
    % VEHICLE DYNAMICS
    % ------------------------------------------------------

    r_act = ...
        r_act + dt*(yaw-r_act)/Tr;

    yawRate = ...
        r_act + sea.Kyaw*lat_slope;

    % -----------------------------------------------------
    % ML ADAPTIVE SPEED
    % ------------------------------------------------------

    u_eff = ...
        desiredSpeed * ...
        max( ...
        0.5, ...
        1-sea.Kres*sea.Hs^2-0.5*abs(pitch_s));

    xDot = ...
        u_eff*cos(heading(k)) + ...
        wx + cx + sea.drift(1);

    yDot = ...
        u_eff*sin(heading(k)) + ...
        wy + cy + sea.drift(2);

    heading(k+1) = ...
        heading(k)+yawRate*dt;

    x(k+1) = ...
        max(2, ...
        min(98,x(k)+xDot*dt));

    y(k+1) = ...
        max(2, ...
        min(98,y(k)+yDot*dt));

    trackingError(k) = ...
        min(hypot( ...
        path(:,1)-x(k+1), ...
        path(:,2)-y(k+1)));

    %% -----------------------------------------------------
    % GRAPHICS
    % ------------------------------------------------------

    update_boat( ...
        boat, ...
        V0, ...
        x(k), ...
        y(k), ...
        0.4+vis_z*eta, ...
        heading(k), ...
        pitch_s, ...
        roll_s);

    for i = 1:numObstacles

        zo = vis_z*sea_waves( ...
            sea, ...
            obstacleX(i), ...
            obstacleY(i), ...
            tk);

        set( ...
            obstacleGraphics(i), ...
            'Vertices', ...
            box_vertices( ...
                obstacleX(i), ...
                obstacleY(i), ...
                obstacleSize(i), ...
                0.65*obstacleSize(i), ...
                2, ...
                zo));

    end

    if mod(k,update_water_every) == 0

        set( ...
            waterSurface, ...
            'ZData', ...
            vis_z*sea_waves( ...
            sea,Xw,Yw,tk));

    end

    set( ...
        trajectory, ...
        'XData',x(1:k), ...
        'YData',y(1:k), ...
        'ZData',ones(1,k));

    %% -----------------------------------------------------
    % WIND ARROWS
    % ------------------------------------------------------

    for i = 1:numel(windArrows)

        set( ...
            windArrows(i), ...
            'UData',wx*40, ...
            'VData',wy*40, ...
            'WData',0);

    end

    %% -----------------------------------------------------
    % CURRENT ARROWS
    % ------------------------------------------------------

    for i = 1:numel(currentArrows)

        [ccx,ccy] = ...
            sea_current( ...
            sea, ...
            cgx(i), ...
            cgy(i), ...
            tk);

        set( ...
            currentArrows(i), ...
            'UData',ccx*40, ...
            'VData',ccy*40, ...
            'WData',0);

    end

    %% -----------------------------------------------------
    % STATUS PANEL
    % ------------------------------------------------------

    if riskLabel == 1

        riskText = 'HIGH RISK';

    else

        riskText = 'SAFE';

    end

    set( ...
        statusText, ...
        'String', ...
        sprintf([ ...
        'USV NAVIGATION STATUS\n' ...
        'Time       : %.1f s\n' ...
        'Distance   : %.2f m\n' ...
        'Track Err. : %.2f m\n' ...
        'Wind       : %.3f m/s\n' ...
        'Current    : %.3f m/s\n' ...
        'Wave Hs    : %.1f m\n' ...
        'Roll/Pitch : %.1f / %.1f deg\n' ...
        'Min Sep.   : %.1f m\n' ...
        'Collisions : %d\n' ...
        'ML Risk    : %s\n' ...
        'Risk Prob. : %.1f %%\n' ...
        'ML Speed   : %.2f m/s\n' ...
        'Look Ahead : %.1f m'], ...
        k*dt, ...
        distanceTarget, ...
        trackingError(k), ...
        windSpeed, ...
        hypot(cx,cy), ...
        sea.Hs, ...
        rad2deg(roll_s), ...
        rad2deg(pitch_s), ...
        minSep, ...
        collisions, ...
        riskText, ...
        100*riskProbability, ...
        desiredSpeed, ...
        currentLookAhead));

    drawnow limitrate;

    %% -----------------------------------------------------
    % VIDEO
    % ------------------------------------------------------

    if save_video && mod(k,2) == 0

        writeVideo( ...
            vw, ...
            getframe(fig));

    end

end

%% =========================================================
% CLOSE VIDEO
% =========================================================

if save_video

    close(vw);

    disp('Video saved as usv_stage4.mp4');

end

if ~reached

    last = N-1;

end

%% =========================================================
% 9. FINAL RESULT
% =========================================================

finalPosition = [x(last) y(last)];

finalError = ...
    norm(target-finalPosition);

mlRiskPercentage = ...
    100*mean(riskLabelLog(1:last));

averageRiskProbability = ...
    100*mean(riskProbabilityLog(1:last));

averageAdaptiveSpeed = ...
    mean(adaptiveSpeedLog(1:last));

averageLookAhead = ...
    mean(adaptiveLookAheadLog(1:last));

%% ---------------------------------------------------------
% SAVE STAGE 4 RESULTS
% ---------------------------------------------------------

stage4_ml_results = struct();

stage4_ml_results.riskProbability = riskProbabilityLog;
stage4_ml_results.riskLabel = riskLabelLog;

stage4_ml_results.adaptiveSpeed = adaptiveSpeedLog;
stage4_ml_results.adaptiveLookAhead = adaptiveLookAheadLog;
stage4_ml_results.avoidanceStrength = avoidanceStrengthLog;

stage4_ml_results.distance = mlDistanceLog;
stage4_ml_results.closingSpeed = mlClosingSpeedLog;
stage4_ml_results.TTC = mlTTClog;

stage4_ml_results.averageRiskProbability = ...
    averageRiskProbability;

stage4_ml_results.riskPercentage = ...
    mlRiskPercentage;

stage4_ml_results.averageSpeed = ...
    averageAdaptiveSpeed;

stage4_ml_results.averageLookAhead = ...
    averageLookAhead;

save( ...
    'stage4_enhanced_ml_results.mat', ...
    'x','y','heading', ...
    'trackingError', ...
    'headingErrLog', ...
    'yawCommand', ...
    'windLog','currentLog', ...
    'heaveLog','rollLog','pitchLog', ...
    'minSepLog', ...
    'collisions', ...
    'reached', ...
    'finalPosition', ...
    'finalError', ...
    'stage4_ml_results', ...
    'sea', ...
    'path', ...
    'target');

%% ---------------------------------------------------------
% RESULTS DISPLAY
% ---------------------------------------------------------

disp(' ');
disp('==============================================');
disp(' STAGE 4 RESULTS');
disp('==============================================');

fprintf( ...
    'Sea state      : %s (Hs = %.1f m)\n', ...
    sea.name,sea.Hs);

fprintf( ...
    'Final Position : (%.2f , %.2f)\n', ...
    finalPosition(1),finalPosition(2));

fprintf( ...
    'Target         : (%.2f , %.2f)\n', ...
    target(1),target(2));

fprintf( ...
    'Final Error    : %.2f m\n', ...
    finalError);

fprintf( ...
    'Travel time    : %.1f s\n', ...
    last*dt);

fprintf( ...
    'Tracking RMSE  : %.2f m\n', ...
    sqrt(mean(trackingError(1:last).^2)));

fprintf( ...
    'Min separation : %.2f m (to moving obstacles)\n', ...
    min(minSepLog(1:last)));

fprintf( ...
    'Collisions     : %d steps\n', ...
    collisions);

fprintf('\n');
fprintf('----------------------------------------------\n');
fprintf(' CUSTOM KNN ML RESULTS\n');
fprintf('----------------------------------------------\n');

fprintf( ...
    'Average risk probability : %.2f %%\n', ...
    averageRiskProbability);

fprintf( ...
    'High-risk steps          : %.2f %%\n', ...
    mlRiskPercentage);

fprintf( ...
    'Average adaptive speed   : %.2f m/s\n', ...
    averageAdaptiveSpeed);

fprintf( ...
    'Average look-ahead       : %.2f m\n', ...
    averageLookAhead);

fprintf('----------------------------------------------\n');

if reached

    disp('STATUS: EMERGENCY LOCATION REACHED');

else

    disp('STATUS: SIMULATION TIME COMPLETED');

end

disp('==============================================');

%% =========================================================
% 10. RESULT PLOTS
% =========================================================

L  = last;
tt = t(1:L);

figure('Name','Stage 4 results');

subplot(4,1,1);

plot( ...
    tt, ...
    trackingError(1:L), ...
    'LineWidth',1.5);

grid on;

ylabel('m');

title( ...
    'Distance from planned path (includes avoidance detours)');

subplot(4,1,2);

plot( ...
    tt, ...
    rad2deg(headingErrLog(1:L)), ...
    'LineWidth',1.2);

hold on;

plot( ...
    tt, ...
    rad2deg(yawCommand(1:L)), ...
    'LineWidth',1.2);

grid on;

ylabel('deg, deg/s');

legend( ...
    'Heading error', ...
    'Yaw-rate command');

subplot(4,1,3);

plot( ...
    tt, ...
    heaveLog(1:L), ...
    'LineWidth',1.2);

hold on;

plot( ...
    tt, ...
    rad2deg(rollLog(1:L)), ...
    'LineWidth',1.2);

plot( ...
    tt, ...
    rad2deg(pitchLog(1:L)), ...
    'LineWidth',1.2);

grid on;

ylabel('m, deg');

legend( ...
    'Heave', ...
    'Roll', ...
    'Pitch');

subplot(4,1,4);

plot( ...
    tt, ...
    minSepLog(1:L), ...
    'LineWidth',1.5);

grid on;

xlabel('Time (s)');
ylabel('m');

title( ...
    'Separation to nearest moving obstacle');

%% =========================================================
% STAGE 4 TRAJECTORY
% =========================================================

figure('Name','Stage 4 trajectory');

plot( ...
    path(:,1), ...
    path(:,2), ...
    'k--', ...
    'LineWidth',1.5);

hold on;

plot( ...
    x(1:L), ...
    y(1:L), ...
    'b', ...
    'LineWidth',2);

plot( ...
    start(1), ...
    start(2), ...
    'go', ...
    'MarkerSize',12, ...
    'LineWidth',3);

plot( ...
    target(1), ...
    target(2), ...
    'rx', ...
    'MarkerSize',15, ...
    'LineWidth',3);

for i = 1:size(rects,1)

    rectangle( ...
        'Position', ...
        [ ...
        rects(i,1)-0.5, ...
        rects(i,3)-0.5, ...
        rects(i,2)-rects(i,1)+1, ...
        rects(i,4)-rects(i,3)+1], ...
        'FaceColor',[0.6 0.6 0.6]);

end

axis equal;
axis([0 100 0 100]);
grid on;

xlabel('X Position (m)');
ylabel('Y Position (m)');

title( ...
    'Planned path vs actual USV trajectory');

legend( ...
    'Planned path', ...
    'USV trajectory', ...
    'Start', ...
    'Incident', ...
    'Location','northwest');

%% =========================================================
% 11. ML RISK PLOT
% =========================================================

figure('Name','Stage 4 ML Risk Prediction');

plot( ...
    tt, ...
    100*riskProbabilityLog(1:L), ...
    'LineWidth',1.5);

hold on;

plot( ...
    tt, ...
    75*ones(size(tt)), ...
    '--', ...
    'LineWidth',1);

plot( ...
    tt, ...
    50*ones(size(tt)), ...
    '--', ...
    'LineWidth',1);

grid on;

xlabel('Time (s)');
ylabel('Collision Risk Probability (%)');

title( ...
    'Custom KNN ML Collision-Risk Prediction');

legend( ...
    'KNN risk probability', ...
    'High-risk threshold', ...
    'Medium-risk threshold', ...
    'Location','northwest');

ylim([0 100]);

disp(' ');
disp('==============================================');
disp(' ENHANCED STAGE 4 + ML COMPLETED');
disp('==============================================');


%% =========================================================
% LOCAL FUNCTION - CUSTOM KNN PREDICTION
% =========================================================

function [label,probability] = knn_predict(model,x)

    % Normalize using the training statistics.

    xNorm = ...
        (x-model.mu)./model.sigma;

    % Euclidean distance from this sample
    % to every training sample.

    distances = ...
        sqrt(sum( ...
        (model.Xtrain-xNorm).^2,2));

    % Sort from nearest to farthest.

    [~,idx] = ...
        sort(distances,'ascend');

    % Number of neighbours.

    K = ...
        min(model.K,length(idx));

    nearest = ...
        idx(1:K);

    % Probability is the percentage of
    % neighbouring samples labelled as risk.

    probability = ...
        mean(model.Ytrain(nearest));

    % Binary prediction.

    label = ...
        probability >= 0.5;

end


%% =========================================================
% LOCAL FUNCTION - USV HULL
% =========================================================

function V = hull_local(L,W,H)

    V = [ ...
         L/2    0   -0.3;
        -L/2 -W/2   -0.3;
         -L/3    0   -0.3;
        -L/2  W/2   -0.3;
         L/2    0    H;
        -L/2 -W/2    H;
        -L/3    0    H;
        -L/2  W/2    H];

end


%% =========================================================
% LOCAL FUNCTION - BOX FACES
% =========================================================

function F = box_faces()

    F = [ ...
        1 2 3 4;
        5 6 7 8;
        1 2 6 5;
        2 3 7 6;
        3 4 8 7;
        4 1 5 8];

end


%% =========================================================
% LOCAL FUNCTION - BOX VERTICES
% =========================================================

function V = box_vertices(x,y,L,W,H,z0)

    V = [ ...
        -L/2 -W/2 0;
         L/2 -W/2 0;
         L/2  W/2 0;
        -L/2  W/2 0;
        -L/2 -W/2 H;
         L/2 -W/2 H;
         L/2  W/2 H;
        -L/2  W/2 H];

    V(:,1) = V(:,1)+x;
    V(:,2) = V(:,2)+y;
    V(:,3) = V(:,3)+z0;

end


%% =========================================================
% LOCAL FUNCTION - UPDATE BOAT
% =========================================================

function update_boat(h,V0,x,y,z,psi,pitch,roll)

    Rz = [ ...
        cos(psi) -sin(psi) 0;
        sin(psi)  cos(psi) 0;
        0         0        1];

    Ry = [ ...
        cos(-pitch) 0 sin(-pitch);
        0           1 0;
       -sin(-pitch) 0 cos(-pitch)];

    Rx = [ ...
        1 0 0;
        0 cos(roll) -sin(roll);
        0 sin(roll)  cos(roll)];

    V = ...
        (Rz*Ry*Rx*V0')';

    V(:,1) = V(:,1)+x;
    V(:,2) = V(:,2)+y;
    V(:,3) = V(:,3)+z;

    set(h,'Vertices',V);

end


%% =========================================================
% LOCAL FUNCTION - STATIC OBSTACLE REPULSION
% =========================================================

function [ax_,ay_] = rect_repulsion(pos,rects,range)

    ax_ = 0;
    ay_ = 0;

    for i = 1:size(rects,1)

        qx = ...
            min(max(pos(1),rects(i,1)-0.5), ...
            rects(i,2)+0.5);

        qy = ...
            min(max(pos(2),rects(i,3)-0.5), ...
            rects(i,4)+0.5);

        d = ...
            hypot(pos(1)-qx,pos(2)-qy);

        if d < range && d > 1e-6

            rep = ...
                (range-d)/range;

            ax_ = ...
                ax_ + ...
                rep*(pos(1)-qx)/d;

            ay_ = ...
                ay_ + ...
                rep*(pos(2)-qy)/d;

        end

    end

end


%% =========================================================
% LOCAL FUNCTION - ANGLE WRAPPING
% =========================================================

function a = wrap_pi(a)

    a = mod(a+pi,2*pi)-pi;

end