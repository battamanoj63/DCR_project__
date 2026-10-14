  clc;
clear;
close all;

%% =========================================================
% STAGE 1 (ENHANCED) - ENVIRONMENT + USV MODEL
% Autonomous USV - Emergency Assistance
%
% Replaces main.m. It now shows:
%   - the water map with obstacles
%   - spatially varying current, gusty wind, irregular waves
%   - OPEN-LOOP behaviour: what happens to a USV that simply points at
%     the incident and has no controller (motivation for Stages 2-4)
% =========================================================

disp('==============================================');
disp(' AUTONOMOUS WATER SURFACE VEHICLE');
disp(' STAGE 1 - ENVIRONMENT + OPEN-LOOP USV MODEL');
disp('==============================================');

%% 1. SETTINGS

dt = 0.1;
T  = 120;
t  = 0:dt:T;
N  = numel(t);

u      = 1.2;               % surge speed (m/s)
start  = [10 10];
target = [90 90];

[map,~,rects] = make_map(4);       %#ok<ASGLU>

presets = {'calm','moderate','rough'};
nS = numel(presets);

%% 2. SEA-STATE PARAMETERS

fprintf('\n%-9s %6s %6s %9s %9s %10s\n', ...
    'Sea','Hs(m)','Tp(s)','Wavelen.','Wind(m/s)','Current');

for s = 1:nS
    sea = make_sea(presets{s},T,dt,s);
    lambda = 9.81*sea.Tp^2/(2*pi);
    fprintf('%-9s %6.1f %6.1f %8.1fm %9.2f %10.2f\n', ...
        sea.name,sea.Hs,sea.Tp,lambda,sea.wind_mean,sea.cur_base);
end

%% 3. OPEN-LOOP SIMULATION
% The boat is pointed at the incident and never corrects its heading.

psi0 = atan2(target(2)-start(2),target(1)-start(1));

traj = cell(1,nS+1);

% (a) ideal case, no disturbances
traj{1} = [start(1)+u*cos(psi0)*t; start(2)+u*sin(psi0)*t]';

% (b) with disturbances
for s = 1:nS

    sea = make_sea(presets{s},T,dt,s);

    x = zeros(1,N); y = zeros(1,N); psi = zeros(1,N);
    x(1) = start(1); y(1) = start(2); psi(1) = psi0;

    for k = 1:N-1

        [wx,wy]   = sea_wind(sea,t(k));
        [cx,cy]   = sea_current(sea,x(k),y(k),t(k));
        [~,dEx,dEy] = sea_waves(sea,x(k),y(k),t(k));

        lat_slope  = -dEx*sin(psi(k)) + dEy*cos(psi(k));
        long_slope =  dEx*cos(psi(k)) + dEy*sin(psi(k));

        u_eff = u*max(0.5,1 - sea.Kres*sea.Hs^2 - 0.5*abs(atan(long_slope)));

        x(k+1)   = x(k) + (u_eff*cos(psi(k)) + wx + cx + sea.drift(1))*dt;
        y(k+1)   = y(k) + (u_eff*sin(psi(k)) + wy + cy + sea.drift(2))*dt;
        psi(k+1) = psi(k) + sea.Kyaw*lat_slope*dt;
    end

    traj{s+1} = [x; y]';
end

%% 4. OPEN-LOOP RESULTS

fprintf('\nOpen-loop miss distance (position after %.0f s, ideal arrival at %.0f s):\n', ...
    T,norm(target-start)/u);

d_ideal = norm(traj{1}(round(norm(target-start)/u/dt)+1,:)-target);
fprintf('  %-9s : %6.2f m\n','Ideal',d_ideal);

for s = 1:nS
    tt = round(norm(target-start)/u/dt)+1;
    fprintf('  %-9s : %6.2f m\n',presets{s}, ...
        norm(traj{s+1}(tt,:)-target));
end

%% 5. FIGURE 1 - ENVIRONMENT

sea = make_sea('rough',T,dt,3);

figure('Name','Stage 1 - Environment');
imagesc(map); set(gca,'YDir','normal'); hold on;
colormap(gca,[1 1 1; 0.6 0.6 0.6]);

[gx,gy] = meshgrid(5:8:95,5:8:95);
[cu,cv] = sea_current(sea,gx,gy,0);
quiver(gx,gy,cu*25,cv*25,0,'b');

th = linspace(0,2*pi,100);
plot(sea.eddy(1)+sea.eddy(3)*cos(th),sea.eddy(2)+sea.eddy(3)*sin(th),'b--');

[wx,wy] = sea_wind(sea,0);
quiver(20,85,wx*40,wy*40,0,'m','LineWidth',2);
text(8,93,'Wind');

d = deg2rad(sea.wave_dir);
quiver(60,88,10*cos(d),10*sin(d),0,'k','LineWidth',2);
text(52,95,'Swell');

plot(start(1),start(2),'go','MarkerSize',10,'LineWidth',3);
plot(target(1),target(2),'kx','MarkerSize',14,'LineWidth',3);
text(start(1)+3,start(2),'USV Start');
text(target(1)-22,target(2)-3,'Incident Location');

axis equal; axis([1 100 1 100]); grid on;
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('Environment (Rough sea): current field, eddy, wind, swell');

%% 6. FIGURE 2 - WAVES

[Xs,Ys] = meshgrid(0:2:100,0:2:100);

figure('Name','Stage 1 - Waves');
subplot(1,2,1);
surf(Xs,Ys,sea_waves(sea,Xs,Ys,0),'EdgeColor','none'); shading interp;
colorbar; view(35,40); axis tight;
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Elevation (m)');
title(sprintf('Sea surface (Hs = %.1f m, Tp = %.0f s)',sea.Hs,sea.Tp));

subplot(1,2,2);
eta_t = zeros(size(t));
for k = 1:N
    eta_t(k) = sea_waves(sea,start(1),start(2),t(k));
end
plot(t,eta_t,'LineWidth',1.2); grid on;
xlabel('Time (s)'); ylabel('Elevation (m)');
title('Wave elevation at the start point');

%% 7. FIGURE 3 - OPEN-LOOP TRAJECTORIES

figure('Name','Stage 1 - Open loop');
imagesc(map); set(gca,'YDir','normal'); hold on;
colormap(gca,[1 1 1; 0.6 0.6 0.6]);

styles = {'k--','g','b','r'};
for i = 1:nS+1
    plot(traj{i}(:,1),traj{i}(:,2),styles{i},'LineWidth',2);
end

plot(start(1),start(2),'o','MarkerSize',12,'LineWidth',3);
plot(target(1),target(2),'x','MarkerSize',15,'LineWidth',3);

axis equal; axis([1 100 1 100]); grid on;
xlabel('X Position (m)'); ylabel('Y Position (m)');
title('Open-loop USV: drift caused by current, wind and waves');
legend('Ideal (no disturbance)','Calm','Moderate','Rough', ...
    'Start','Incident','Location','northwest');

disp(' ');
disp('==============================================');
disp(' STAGE 1 COMPLETED SUCCESSFULLY');
disp('==============================================');
