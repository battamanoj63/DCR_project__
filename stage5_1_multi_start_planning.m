clc;
clear;
close all;

%% =========================================================
% STAGE 5.1 - MULTI-START, CURRENT-AWARE A* PATH PLANNING
% Autonomous USV - Emergency Assistance
%
% Several rescue boats (bases) start from DIFFERENT points and must reach the
% same incident.  For every start point the Stage-2 current-aware A* is run:
%   1. in the normal planning sea ('moderate')           -> baseline plans
%   2. in every rougher sea state (calm ... severe)      -> how the plan changes
%   3. with the MOVING objects of the ocean (ships, floating debris, patrol
%      boat) predicted ahead in time                      -> dynamic-aware plans
% and the fleet is ranked by estimated time of arrival (ETA).
%
% Needs (your existing files): make_map.m, sea_current.m, sea_waves.m, sea_wind.m
% Needs (this package)       : stage5_support\*.m
% Output                     : multi_start_plan.mat  + 4 figures
% =========================================================

disp('==============================================');
disp(' STAGE 5.1 - MULTI-START A* PATH PLANNING');
disp('==============================================');

here = fileparts(mfilename('fullpath'));
if isempty(here), here = pwd; end
addpath(here); addpath(fullfile(here,'stage5_support'));
need = {'make_map','sea_current','sea_waves','sea_wind'};
for q = 1:numel(need)
    if exist(need{q},'file') == 0
        error('%s.m not found. Put this package in the folder of your Stage 1-4 files.',need{q});
    end
end

%% =========================================================
% 1. SETTINGS
% =========================================================

cfg  = stage5_config();            % starts, target, sea list, planner settings
P    = build_plan_context(cfg);    % maps, clearance, 2 m planning grid

main_sea = 'moderate';             % sea state the baseline plans are made for
ub       = cfg.u_nom;              % planning speed command (m/s)

starts   = cfg.starts;   nS = size(starts,1);
target   = cfg.target;
seaList  = cfg.seas;     nSea = numel(seaList);

cStart = lines(nS);                                    % one colour per start
cSea   = [0.20 0.55 0.90; 0.15 0.65 0.30; 0.95 0.70 0.10; 0.90 0.30 0.10; 0.55 0.05 0.55];

noMoving = struct('m',0,'pos',zeros(0,2),'vel',zeros(0,2),'r',zeros(0,1));

fprintf('\nStart points (%d boats) -> incident at (%d,%d)\n',nS,target(1),target(2));
for i = 1:nS
    fprintf('  %d  %-18s straight-line distance %.1f m\n', ...
        i,cfg.start_names{i},norm(target-starts(i,:)));
end

%% =========================================================
% 2. SEA STATES
% =========================================================

seas = cell(1,nSea);
for j = 1:nSea
    seas{j} = make_sea_ext(seaList{j},cfg.T,cfg.dt,cfg.seed);
end
jMain = find(strcmpi(seaList,main_sea),1);
if isempty(jMain), jMain = 1; end

fprintf('\n%-9s %6s %6s %10s %9s %10s\n','Sea','Hs(m)','Tp(s)','Wind(m/s)','Current','Eddies');
for j = 1:nSea
    s = seas{j};
    fprintf('%-9s %6.1f %6.1f %10.2f %9.2f %10d\n', ...
        s.name,s.Hs,s.Tp,s.wind_mean,s.cur_base,1+size(s.eddies,1));
end

%% =========================================================
% 3. A* FOR EVERY START IN EVERY SEA STATE (no moving objects)
% =========================================================

paths = cell(nS,nSea);
eta   = zeros(nS,nSea);
len   = zeros(nS,nSea);
clr   = zeros(nS,nSea);

for j = 1:nSea
    E = make_plan_field(P,cfg,seas{j},0,ub);
    for i = 1:nS
        [pth,ok] = plan_path(P,cfg,starts(i,:),noMoving,E,false);
        if ~ok
            error('No safe path from start %d in sea %s.',i,seaList{j});
        end
        paths{i,j} = pth;
        [eta(i,j),len(i,j)] = path_eta(P,E,pth);
        ix = min(100,max(1,round(pth(:,1))));  iy = min(100,max(1,round(pth(:,2))));
        clr(i,j) = min(P.clear_d(sub2ind(size(P.clear_d),iy,ix)));
    end
end

%% =========================================================
% 4. RESULTS - BASELINE PLANS
% =========================================================

fprintf('\n====================================================================\n');
fprintf(' BASELINE PLANS (%s sea)\n',seas{jMain}.name);
fprintf('====================================================================\n');
fprintf(' %-18s %9s %9s %8s %9s %9s\n','Start','Straight','Path(m)','ETA(s)','MinClr(m)','Rank');
[~,ord] = sort(eta(:,jMain));
rank = zeros(nS,1);  rank(ord) = 1:nS;
for i = 1:nS
    fprintf(' %-18s %9.1f %9.1f %8.1f %9.2f %9d\n',cfg.start_names{i}, ...
        norm(target-starts(i,:)),len(i,jMain),eta(i,jMain),clr(i,jMain),rank(i));
end
fprintf('--------------------------------------------------------------------\n');
fprintf(' First boat on scene : %s (ETA %.1f s)\n',cfg.start_names{ord(1)},eta(ord(1),jMain));
fprintf(' Last boat on scene  : %s (ETA %.1f s)\n',cfg.start_names{ord(end)},eta(ord(end),jMain));

fprintf('\n ESTIMATED ETA (s) OF EVERY START IN EVERY SEA STATE (plan made for that sea)\n');
fprintf(' %-18s',''); fprintf(' %9s',seaList{:}); fprintf('\n');
for i = 1:nS
    fprintf(' %-18s',cfg.start_names{i}); fprintf(' %9.1f',eta(i,:)); fprintf('\n');
end

fprintf('\n PATH LENGTH (m)\n');
fprintf(' %-18s',''); fprintf(' %9s',seaList{:}); fprintf('\n');
for i = 1:nS
    fprintf(' %-18s',cfg.start_names{i}); fprintf(' %9.1f',len(i,:)); fprintf('\n');
end

%% =========================================================
% 5. DYNAMIC-AWARE PLANS (moving ships / debris predicted ahead)
% =========================================================

t   = 0:cfg.dt:cfg.T;
tr  = make_traffic(seas{jMain},cfg,t);
W0  = struct('m',tr.m,'pos',tr.pos(:,:,1),'vel',tr.vel(:,:,1),'r',tr.r);

Em  = make_plan_field(P,cfg,seas{jMain},0,ub);
pathDyn = cell(nS,1);  etaDyn = zeros(nS,1);  lenDyn = zeros(nS,1);  relaxDyn = zeros(nS,1);
for i = 1:nS
    [pth,ok,info] = plan_path(P,cfg,starts(i,:),W0,Em,true);
    if ~ok, error('No dynamic-aware path from start %d.',i); end
    pathDyn{i} = pth;  relaxDyn(i) = info.relax;
    [etaDyn(i),lenDyn(i)] = path_eta(P,Em,pth);
end

fprintf('\n====================================================================\n');
fprintf(' DYNAMIC-AWARE PLANS (%d moving objects predicted %d s ahead)\n',tr.m,cfg.plan_H);
fprintf('====================================================================\n');
fprintf(' %-18s %10s %10s %10s %10s\n','Start','Static(m)','Dynamic(m)','Extra(m)','Extra(s)');
for i = 1:nS
    fprintf(' %-18s %10.1f %10.1f %10.1f %10.1f\n',cfg.start_names{i}, ...
        len(i,jMain),lenDyn(i),lenDyn(i)-len(i,jMain),etaDyn(i)-eta(i,jMain));
end

%% =========================================================
% 6. FIGURE 1 - ALL START POINTS, BASELINE PLANS
% =========================================================

sea = seas{jMain};

f1 = figure('Name','Stage 5.1 - Multi-start A*','Color','w');
ax = axes(f1);  draw_map(ax,P);
[gx,gy] = meshgrid(5:8:95,5:8:95);
[cu,cv] = sea_current_ext(sea,gx,gy,0);
quiver(ax,gx,gy,cu*25,cv*25,0,'Color',[0.55 0.65 0.95]);
th = linspace(0,2*pi,100);
plot(ax,sea.eddy(1)+sea.eddy(3)*cos(th),sea.eddy(2)+sea.eddy(3)*sin(th),'--','Color',[0.4 0.5 0.9]);
hl = gobjects(1,nS);
for i = 1:nS
    hl(i) = plot(ax,paths{i,jMain}(:,1),paths{i,jMain}(:,2),'-','Color',cStart(i,:),'LineWidth',2.5);
    plot(ax,starts(i,1),starts(i,2),'o','MarkerSize',11,'LineWidth',2.5,'Color',cStart(i,:), ...
        'MarkerFaceColor','w');
    text(ax,starts(i,1)+2,starts(i,2)+3,sprintf('%d',i),'Color',cStart(i,:),'FontWeight','bold');
end
plot(ax,target(1),target(2),'kp','MarkerSize',16,'MarkerFaceColor','y','LineWidth',1.5);
text(ax,target(1)-26,target(2)+4,'Incident Location','FontWeight','bold');
xlabel(ax,'X Position (m)'); ylabel(ax,'Y Position (m)');
title(ax,sprintf('Multi-start current-aware A*  (%s sea, arrows = current)',sea.name));
legend(ax,hl,cfg.start_names,'Location','southoutside','Orientation','horizontal');

%% =========================================================
% 7. FIGURE 2 - HOW THE PLAN CHANGES WITH THE SEA STATE
% =========================================================

f2 = figure('Name','Stage 5.1 - Plans in different seas','Color','w','Position',[80 60 1200 750]);
for i = 1:nS
    ax = subplot(2,3,i);  draw_map(ax,P,false);
    hh = gobjects(1,nSea);
    for j = 1:nSea
        hh(j) = plot(ax,paths{i,j}(:,1),paths{i,j}(:,2),'-','Color',cSea(j,:),'LineWidth',1.8);
    end
    plot(ax,starts(i,1),starts(i,2),'go','MarkerSize',9,'LineWidth',2.5);
    plot(ax,target(1),target(2),'kp','MarkerSize',12,'MarkerFaceColor','y');
    title(ax,cfg.start_names{i},'FontSize',9);
    if i == 1, legend(ax,hh,seaList,'Location','northwest','FontSize',7); end
end
ax = subplot(2,3,6);
bar(ax,eta);  grid(ax,'on');
set(ax,'XTick',1:nS);
xlabel(ax,'Start point #'); ylabel(ax,'Estimated ETA (s)');
title(ax,'ETA of the plan in each sea state','FontSize',9);
legend(ax,seaList,'Location','northwest','FontSize',7);

%% =========================================================
% 8. FIGURE 3 - STATIC PLAN vs DYNAMIC-AWARE PLAN
% =========================================================

f3 = figure('Name','Stage 5.1 - Dynamic-aware plans','Color','w');
ax = axes(f3);  draw_map(ax,P);
for a = 1:tr.m                                         % predicted tracks of moving objects
    tau = linspace(0,cfg.plan_H,30);
    px  = W0.pos(a,1)+W0.vel(a,1)*tau;  py = W0.pos(a,2)+W0.vel(a,2)*tau;
    plot(ax,px,py,'r:','LineWidth',1.5);
    rectangle(ax,'Position',[W0.pos(a,1)-tr.r(a) W0.pos(a,2)-tr.r(a) 2*tr.r(a) 2*tr.r(a)], ...
        'Curvature',[1 1],'FaceColor',[0.85 0.2 0.15],'EdgeColor','k');
    rectangle(ax,'Position',[px(end)-tr.r(a) py(end)-tr.r(a) 2*tr.r(a) 2*tr.r(a)], ...
        'Curvature',[1 1],'EdgeColor',[0.85 0.2 0.15],'LineStyle','--');
end
for i = 1:nS
    plot(ax,paths{i,jMain}(:,1),paths{i,jMain}(:,2),'--','Color',cStart(i,:),'LineWidth',1.2);
    plot(ax,pathDyn{i}(:,1),pathDyn{i}(:,2),'-','Color',cStart(i,:),'LineWidth',2.6);
    plot(ax,starts(i,1),starts(i,2),'o','MarkerSize',10,'LineWidth',2.5,'Color',cStart(i,:));
end
plot(ax,target(1),target(2),'kp','MarkerSize',16,'MarkerFaceColor','y');
xlabel(ax,'X Position (m)'); ylabel(ax,'Y Position (m)');
title(ax,'Dashed: plan ignoring moving objects   Solid: plan with predicted ship / debris motion');

%% =========================================================
% 9. FIGURE 4 - FLEET SUMMARY
% =========================================================

f4 = figure('Name','Stage 5.1 - Fleet summary','Color','w');
subplot(1,2,1);
bar([len(:,jMain) lenDyn]);  grid on;
set(gca,'XTick',1:nS);
xlabel('Start point #'); ylabel('Path length (m)');
legend('Static plan','Dynamic-aware plan','Location','northwest');
title('Path length per start');
subplot(1,2,2);
bar([eta(:,jMain) etaDyn]);  grid on;
set(gca,'XTick',1:nS);
xlabel('Start point #'); ylabel('Estimated ETA (s)');
legend('Static plan','Dynamic-aware plan','Location','northwest');
title('Arrival time per start');

%% =========================================================
% 10. SAVE
% =========================================================

multi_start = struct();
multi_start.starts     = starts;
multi_start.target     = target;
multi_start.names      = cfg.start_names;
multi_start.seaList    = seaList;
multi_start.paths      = paths;        % {start, sea}
multi_start.eta        = eta;
multi_start.length     = len;
multi_start.pathDyn    = pathDyn;
multi_start.etaDyn     = etaDyn;
save('multi_start_plan.mat','multi_start');

disp(' ');
disp('Saved multi_start_plan.mat');
disp('==============================================');
disp(' STAGE 5.1 COMPLETED');
disp('==============================================');
