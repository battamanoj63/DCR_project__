function P = build_plan_context(cfg)
% BUILD_PLAN_CONTEXT  Maps, clearance and the coarse planning grid.
%   P.map, P.safe_map, P.clear_d : 100x100, index as (y,x)
%   P.xs                         : grid coordinates (2:2:100)
%   P.blk, P.pen                 : blocked flag / clearance penalty on the grid

    n = cfg.map_size;
    [map,safe_map] = make_map(cfg.safety_margin);

    % start points and incident must be usable water
    for i = 1:size(cfg.starts,1)
        safe_map(cfg.starts(i,2),cfg.starts(i,1)) = 0;
    end
    safe_map(cfg.target(2),cfg.target(1)) = 0;

    % distance to the nearest obstacle cell
    [oy,ox] = find(map == 1);
    clear_d = zeros(n,n);
    for yy = 1:n
        for xx = 1:n
            clear_d(yy,xx) = min(hypot(ox-xx,oy-yy));
        end
    end
    penalty = max(0,(cfg.comfort_dist-clear_d)/cfg.comfort_dist).^2;

    xs = cfg.grid_step:cfg.grid_step:n;          % 2,4,...,100

    P.n        = n;
    P.map      = map;
    P.safe_map = safe_map;
    P.clear_d  = clear_d;
    P.penalty  = penalty;
    P.xs       = xs;
    P.blk      = safe_map(xs,xs) == 1;           % (iy,ix) on the planning grid
    P.pen      = penalty(xs,xs);
end
