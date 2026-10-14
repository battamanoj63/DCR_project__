function cfg = stage5_config()
% STAGE5_CONFIG  Settings for Stage 5.1 (multi-start, current-aware A*).
%   All start points use EVEN coordinates so they sit exactly on the 2 m
%   planning grid.  Starts and target are in free water (outside the
%   inflated obstacles of make_map).

    % ---- simulation
    cfg.dt     = 0.1;          % s
    cfg.T      = 300;          % s
    cfg.seed   = 1;            % random seed of the sea states

    % ---- boat
    cfg.u_nom   = 1.2;         % m/s  commanded speed
    cfg.boat_r  = 1.0;         % m    boat radius used for avoidance
    cfg.thresh  = 2.0;         % m    "arrived" distance to the incident

    % ---- scenario
    cfg.target = [90 90];
    cfg.starts = [10 10;       % 1
                   8 56;       % 2
                  46  8;       % 3
                  92 50;       % 4
                  36 92];      % 5   (max 5 starts: Figure 2 has 2x3 panels)
    cfg.start_names = {'South-West Base','West Base','South Base', ...
                       'East Base','North Base'};

    % ---- sea states (make_sea_ext knows all of these)
    cfg.seas = {'calm','moderate','rough','heavy','severe'};

    % ---- map and planner
    cfg.map_size      = 100;
    cfg.safety_margin = 4;     % cells
    cfg.grid_step     = 2;     % m   A* grid spacing
    cfg.comfort_dist  = 10;    % m   clearance penalty distance
    cfg.w_clear       = 3;     % clearance penalty weight
    cfg.plan_H        = 60;    % s   prediction horizon for moving objects
    cfg.dyn_margin    = 2;     % m   extra clearance around moving objects
end
