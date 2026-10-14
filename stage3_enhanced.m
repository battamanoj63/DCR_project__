clc;
clear;
close all;

%% =========================================================
% STAGE 3 (ENHANCED) - PID PATH FOLLOWING IN A REALISTIC SEA
% Autonomous USV - Emergency Assistance
%
% ORIGINAL STAGE 3 + CUSTOM KNN ML
%
% KNN is used only for:
%   - collision-risk prediction
%   - adaptive speed
%   - adaptive lookahead
%
% No external dataset.
% No Statistics and Machine Learning Toolbox.
% =========================================================

disp('==============================================');
disp(' STAGE 3 ENHANCED - A* + PID + CURRENT + WAVES + ML');
disp('==============================================');

%% =========================================================
% 1. LOAD PATH AND MAP
% ==========================================================

if ~isfile('astar_path.mat')
    error('astar_path.mat not found. Run stage2_astar_enhanced_knn.m first.');
end

if ~isfile('knn_risk_model.mat')
    error('knn_risk_model.mat not found. Run stage2_astar_enhanced_knn.m first.');
end

S = load('astar_path.mat');

path = S.path;

[map,safe_map] = make_map(4);

M = load('knn_risk_model.mat');

knnModel = M.knnModel;

%% =========================================================
% 2. CONFIGURATION
% ==========================================================

cfg.dt            = 0.1;
cfg.T             = 250;
cfg.u             = 1.2;
cfg.max_yaw_rate  = 0.8;
cfg.Tr            = 0.5;

cfg.Kp = 2.5;
cfg.Ki = 0.015;
cfg.Kd = 0.35;

cfg.lookahead     = 4;
cfg.thresh        = 2.0;

cfg.gps_sigma     = 0.3;
cfg.gps_rate      = 1;
cfg.compass_sigma = deg2rad(1);

%% =========================================================
% 3. RUN THE THREE SEA STATES
% ==========================================================

presets = {'calm','moderate','rough'};

nS = numel(presets);

seas = cell(1,nS);
results = cell(1,nS);

for s = 1:nS

    fprintf('\nRunning scenario: %s ...\n',presets{s});

    seas{s} = make_sea( ...
        presets{s},cfg.T,cfg.dt,s);

    results{s} = run_usv( ...
        path,map,safe_map,cfg,seas{s},s,knnModel);

end

%% =========================================================
% 4. RESULTS TABLE
% ==========================================================

fprintf('\n=====================================================================\n');

fprintf(' %-9s %7s %7s %7s %8s %8s %8s %6s\n', ...
    'Sea','RMSE','MAE','Max','FinalErr', ...
    'Time(s)','MinClr','Hits');

fprintf('=====================================================================\n');

for s = 1:nS

    r = results{s};

    fprintf(' %-9s %7.3f %7.3f %7.3f %8.3f %8.1f %8.2f %6d\n', ...
        seas{s}.name, ...
        r.RMSE, ...
        r.MAE, ...
        r.max_err, ...
        r.final_err, ...
        r.travel_time, ...
        r.min_clearance, ...
        r.collisions);

end

fprintf('=====================================================================\n');

fprintf('Hits = steps inside a real obstacle cell. MinClr = closest approach (m).\n');

save('stage3_enhanced_results.mat', ...
    'results','seas','cfg','path');

%% =========================================================
% 5. FIGURE 1 - ENVIRONMENT
% ==========================================================

sea = seas{3};

figure('Name','Environment');

imagesc(map);

set(gca,'YDir','normal');

hold on;

colormap(gca,[1 1 1; 0.6 0.6 0.6]);

[gx,gy] = meshgrid(5:8:95,5:8:95);

[cu,cv] = sea_current(sea,gx,gy,0);

quiver(gx,gy,cu*25,cv*25,0,'b');

plot(path(:,1),path(:,2), ...
    'r','LineWidth',2);

plot(path(1,1),path(1,2), ...
    'go','MarkerSize',10,'LineWidth',3);

plot(path(end,1),path(end,2), ...
    'kx','MarkerSize',14,'LineWidth',3);

th = linspace(0,2*pi,100);

plot( ...
    sea.eddy(1)+sea.eddy(3)*cos(th), ...
    sea.eddy(2)+sea.eddy(3)*sin(th), ...
    'b--');

d = deg2rad(sea.wave_dir);

quiver(12,88, ...
    10*cos(d),10*sin(d), ...
    0,'k','LineWidth',2);

text(8,94,'Swell direction');

axis equal;
axis([1 100 1 100]);
grid on;

xlabel('X Position (m)');
ylabel('Y Position (m)');

title('Rough sea: current field (blue arrows), eddy (dashed), planned path');

%% =========================================================
% 6. FIGURE 2 - WAVE SURFACE + SPECTRUM
% ==========================================================

[Xs,Ys] = meshgrid(0:2:100,0:2:100);

Eta = sea_waves(sea,Xs,Ys,0);

figure('Name','Waves');

subplot(1,2,1);

surf(Xs,Ys,Eta,'EdgeColor','none');

shading interp;

colorbar;

view(35,40);

axis tight;

xlabel('X (m)');
ylabel('Y (m)');
zlabel('Elevation (m)');

title(sprintf( ...
    'Sea surface snapshot (Hs=%.1f m, Tp=%.0f s)', ...
    sea.Hs,sea.Tp));

subplot(1,2,2);

wp = 2*pi/sea.Tp;

w_plot = linspace(0.3*wp,4*wp,300);

S_plot = ...
    (5/16)*sea.Hs^2*wp^4 ./ w_plot.^5 .* ...
    exp(-1.25*(wp./w_plot).^4);

plot(w_plot,S_plot,'LineWidth',2);

grid on;

xlabel('\omega (rad/s)');
ylabel('S(\omega) (m^2 s/rad)');

title('Wave spectrum (Bretschneider)');

%% =========================================================
% 7. FIGURE 3 - TRAJECTORIES
% ==========================================================

figure('Name','Trajectories');

imagesc(map);

set(gca,'YDir','normal');

hold on;

colormap(gca,[1 1 1; 0.6 0.6 0.6]);

plot(path(:,1),path(:,2), ...
    'k--','LineWidth',1.5);

cols = {'g','b','r'};

for s = 1:nS

    r = results{s};

    plot( ...
        r.x(1:r.last), ...
        r.y(1:r.last), ...
        cols{s}, ...
        'LineWidth',2);

end

plot(path(1,1),path(1,2), ...
    'o','MarkerSize',12,'LineWidth',3);

plot(path(end,1),path(end,2), ...
    'x','MarkerSize',15,'LineWidth',3);

axis equal;
axis([1 100 1 100]);
grid on;

xlabel('X Position (m)');
ylabel('Y Position (m)');

title('Planned path vs PID trajectory in different sea states');

legend('Planned path','Calm','Moderate','Rough', ...
    'Start','Incident','Location','northwest');

%% =========================================================
% 8. FIGURE 4 - TIME HISTORIES
% ==========================================================

r = results{3};

L = r.last;

tt = r.t(1:L);

figure('Name','Rough sea time histories');

subplot(4,1,1);

plot(tt,r.tracking_error(1:L), ...
    'LineWidth',1.5);

grid on;

ylabel('Cross-track (m)');
title('Tracking error');

subplot(4,1,2);

plot(tt,rad2deg(r.heading_error(1:L)), ...
    'LineWidth',1.2);

hold on;

plot(tt,rad2deg(r.yaw_cmd(1:L)), ...
    'LineWidth',1.2);

grid on;

ylabel('deg, deg/s');

legend('Heading error','Yaw-rate command');

subplot(4,1,3);

plot(tt,r.heave(1:L), ...
    'LineWidth',1.2);

hold on;

plot(tt,rad2deg(r.roll(1:L)), ...
    'LineWidth',1.2);

plot(tt,rad2deg(r.pitch(1:L)), ...
    'LineWidth',1.2);

grid on;

ylabel('m, deg');

legend('Heave','Roll','Pitch');

subplot(4,1,4);

plot(tt,hypot(r.wind_x(1:L),r.wind_y(1:L)), ...
    'LineWidth',1.2);

hold on;

plot(tt,hypot(r.cur_x(1:L),r.cur_y(1:L)), ...
    'LineWidth',1.2);

plot(tt,r.u_eff(1:L), ...
    'LineWidth',1.2);

grid on;

xlabel('Time (s)');
ylabel('m/s');

legend('Wind drift','Current','Effective surge speed');

%% =========================================================
% 9. FIGURE 5 - COMPARISON
% ==========================================================

figure('Name','Comparison');

vals = [ ...
    cellfun(@(q) q.RMSE,results); ...
    cellfun(@(q) q.max_err,results)]';

bar(vals);

grid on;

set(gca,'XTickLabel', ...
    cellfun(@(q) q.name,seas,'UniformOutput',false));

ylabel('Error (m)');

legend('RMSE','Max error', ...
    'Location','northwest');

title('Path-tracking error vs sea state');

disp(' ');
disp('==============================================');
disp(' ENHANCED STAGE 3 + KNN COMPLETED');
disp('==============================================');


%% =========================================================
% LOCAL FUNCTION
% ==========================================================

function res = run_usv( ...
    path,map,safe_map,cfg,sea,seed,knnModel)

    rng(seed);

    dt = cfg.dt;

    t = 0:dt:cfg.T;

    N = numel(t);

    map_size = size(map,1);

    target = path(end,:);

    %% Original state variables

    x = zeros(1,N);
    y = zeros(1,N);
    psi = zeros(1,N);

    x(1) = path(1,1);
    y(1) = path(1,2);

    k5 = min(5,size(path,1));

    psi(1) = atan2( ...
        path(k5,2)-y(1), ...
        path(k5,1)-x(1));

    r_act = 0;

    tracking_error = zeros(1,N);
    heading_error = zeros(1,N);

    yaw_cmd = zeros(1,N);

    heave = zeros(1,N);
    roll = zeros(1,N);
    pitch = zeros(1,N);

    wind_x = zeros(1,N);
    wind_y = zeros(1,N);

    cur_x = zeros(1,N);
    cur_y = zeros(1,N);

    u_eff_log = zeros(1,N);

    %% =====================================================
    % ONLY NEW ML LOGS
    % ======================================================

    ml_risk = zeros(1,N);
    ml_probability = zeros(1,N);
    ml_speed = zeros(1,N);
    ml_lookahead = zeros(1,N);

    %% Original variables

    integral_error = 0;
    prev_error = 0;
    d_filt = 0;
    idx = 1;

    meas_pos = [x(1) y(1)];

    gps_every = max(1,round(1/(cfg.gps_rate*dt)));

    [obs_y,obs_x] = find(map == 1);

    collisions = 0;

    last = N;

    reached = false;

    %% =====================================================
    % MAIN LOOP
    % ======================================================

    for k = 1:N-1

        %% Target reached

        if norm(target-[x(k) y(k)]) <= cfg.thresh

            reached = true;
            last = k;

            x(k+1:end) = x(k);
            y(k+1:end) = y(k);
            psi(k+1:end) = psi(k);

            break;

        end

        %% Sensors

        if mod(k-1,gps_every) == 0

            meas_pos = ...
                [x(k) y(k)] + ...
                cfg.gps_sigma*randn(1,2);

        end

        psi_meas = ...
            psi(k) + ...
            cfg.compass_sigma*randn;

        %% =================================================
        % ORIGINAL GUIDANCE
        % ==================================================

        hi = min(size(path,1),idx+15);

        dd = hypot( ...
            path(idx:hi,1)-meas_pos(1), ...
            path(idx:hi,2)-meas_pos(2));

        [~,m] = min(dd);

        idx = idx+m-1;

        %% =================================================
        % ML COLLISION-RISK PREDICTION
        % ==================================================

        % Distance to nearest real obstacle

        obstacleDistance = min( ...
            hypot(obs_x-x(k),obs_y-y(k)));

        %% Approximate closing speed

        if k > 1

            currentVelocity = ...
                hypot( ...
                x(k)-x(k-1), ...
                y(k)-y(k-1))/dt;

        else

            currentVelocity = cfg.u;

        end

        closingSpeed = max(0,currentVelocity);

        %% Environment values

        tk = t(k);

        [wind_x(k),wind_y(k)] = ...
            sea_wind(sea,tk);

        [cur_x(k),cur_y(k)] = ...
            sea_current(sea,x(k),y(k),tk);

        windSpeed = hypot(wind_x(k),wind_y(k));

        currentSpeed = hypot(cur_x(k),cur_y(k));

        %% Current desired direction

        tempHi = min(size(path,1),idx+5);

        desiredPathHeading = atan2( ...
            path(tempHi,2)-meas_pos(2), ...
            path(tempHi,1)-meas_pos(1));

        headingErrorML = rad2deg( ...
            wrap_pi(desiredPathHeading-psi_meas));

        relativeAngle = headingErrorML;

        %% TTC

        if closingSpeed > 0.05

            TTC = obstacleDistance/closingSpeed;

        else

            TTC = 60;

        end

        TTC = min(TTC,60);

        %% KNN features

        mlFeatures = [ ...
            obstacleDistance
            closingSpeed
            relativeAngle
            windSpeed
            currentSpeed
            headingErrorML
            TTC];

        [riskLabel,riskProbability] = ...
            knn_predict(knnModel,mlFeatures);

        ml_risk(k) = riskLabel;

        ml_probability(k) = riskProbability;

        %% Adaptive speed/lookahead

        if riskProbability >= 0.75

            desiredSpeed = 0.7;
            adaptiveLookahead = 9;

        elseif riskProbability >= 0.50

            desiredSpeed = 0.9;
            adaptiveLookahead = 7;

        else

            desiredSpeed = cfg.u;
            adaptiveLookahead = cfg.lookahead;

        end

        ml_speed(k) = desiredSpeed;

        ml_lookahead(k) = adaptiveLookahead;

        %% =================================================
        % ORIGINAL GUIDANCE WITH ONLY LOOKAHEAD ADAPTED
        % ==================================================

        li = idx;
        acc = 0;

        while li < size(path,1) && ...
                acc < adaptiveLookahead

            acc = acc + ...
                norm(path(li+1,:)-path(li,:));

            li = li+1;

        end

        wp = path(li,:);

        des = atan2( ...
            wp(2)-meas_pos(2), ...
            wp(1)-meas_pos(1));

        e = wrap_pi(des-psi_meas);

        heading_error(k) = e;

        %% Original PID

        integral_error = ...
            max(min(integral_error+e*dt,5),-5);

        d_raw = ...
            (e-prev_error)/dt;

        d_filt = ...
            0.8*d_filt + ...
            0.2*d_raw;

        prev_error = e;

        cmd = ...
            cfg.Kp*e + ...
            cfg.Ki*integral_error + ...
            cfg.Kd*d_filt;

        cmd = max( ...
            min(cmd,cfg.max_yaw_rate), ...
            -cfg.max_yaw_rate);

        yaw_cmd(k) = cmd;

        %% =================================================
        % ORIGINAL ENVIRONMENT
        % ==================================================

        [eta,dEx,dEy] = ...
            sea_waves(sea,x(k),y(k),tk);

        long_slope = ...
            dEx*cos(psi(k)) + ...
            dEy*sin(psi(k));

        lat_slope = ...
            -dEx*sin(psi(k)) + ...
            dEy*cos(psi(k));

        heave(k) = eta;

        pitch(k) = atan(long_slope);

        roll(k) = atan(lat_slope);

        %% ONLY CHANGE:
        % Original cfg.u replaced by ML-adaptive desiredSpeed

        u_eff = desiredSpeed * ...
            max(0.5, ...
            1 - sea.Kres*sea.Hs^2 ...
            - 0.5*abs(pitch(k)));

        u_eff_log(k) = u_eff;

        yaw_dist = sea.Kyaw*lat_slope;

        %% Original vehicle model

        r_act = ...
            r_act + ...
            dt*(cmd-r_act)/cfg.Tr;

        psi_dot = r_act + yaw_dist;

        x_dot = ...
            u_eff*cos(psi(k)) + ...
            wind_x(k) + ...
            cur_x(k) + ...
            sea.drift(1);

        y_dot = ...
            u_eff*sin(psi(k)) + ...
            wind_y(k) + ...
            cur_y(k) + ...
            sea.drift(2);

        x(k+1) = ...
            min(map_size, ...
            max(1,x(k)+x_dot*dt));

        y(k+1) = ...
            min(map_size, ...
            max(1,y(k)+y_dot*dt));

        psi(k+1) = ...
            psi(k) + psi_dot*dt;

        %% Original collision check

        if map(round(y(k+1)),round(x(k+1))) == 1

            collisions = collisions+1;

        end

    end

    if ~reached

        last = N;

    end

    %% Original tracking error

    for k = 1:last

        tracking_error(k) = ...
            min(hypot( ...
            path(:,1)-x(k), ...
            path(:,2)-y(k)));

    end

    %% Original minimum clearance

    min_clear = inf;

    for k = 1:5:last

        min_clear = min( ...
            min_clear, ...
            min(hypot( ...
            obs_x-x(k), ...
            obs_y-y(k))));

    end

    %% Results

    res.t = t;

    res.x = x;
    res.y = y;
    res.psi = psi;

    res.last = last;

    res.tracking_error = tracking_error;
    res.heading_error = heading_error;

    res.yaw_cmd = yaw_cmd;

    res.heave = heave;
    res.roll = roll;
    res.pitch = pitch;

    res.wind_x = wind_x;
    res.wind_y = wind_y;

    res.cur_x = cur_x;
    res.cur_y = cur_y;

    res.u_eff = u_eff_log;

    %% ONLY NEW ML RESULTS

    res.ml_risk = ml_risk;
    res.ml_probability = ml_probability;
    res.ml_speed = ml_speed;
    res.ml_lookahead = ml_lookahead;

    %% Original metrics

    res.RMSE = ...
        sqrt(mean(tracking_error(1:last).^2));

    res.MAE = ...
        mean(tracking_error(1:last));

    res.max_err = ...
        max(tracking_error(1:last));

    res.final_err = ...
        norm(target-[x(last) y(last)]);

    res.travel_time = t(last);

    res.reached = reached;

    res.collisions = collisions;

    res.min_clearance = min_clear;

end


%% =========================================================
% KNN PREDICTION FUNCTION
% ==========================================================

function [label,probability] = ...
    knn_predict(model,x)

    xNorm = ...
        (x(:)' - model.mu) ./ model.sigma;

    distances = sqrt( ...
        sum((model.Xtrain-xNorm).^2,2));

    [~,idx] = ...
        sort(distances,'ascend');

    K = min(model.K,length(idx));

    nearest = idx(1:K);

    probability = ...
        mean(model.Ytrain(nearest));

    label = probability >= 0.5;

end


%% =========================================================
% WRAP ANGLE
% ==========================================================

function a = wrap_pi(a)

    a = mod(a+pi,2*pi)-pi;

end