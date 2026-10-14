function [path,ok,info] = plan_path(P,cfg,start,W,E,dynamic)
% PLAN_PATH  Safe, current-aware A* from START to cfg.target on the 2 m grid.
%   P       planning context (build_plan_context)
%   W       moving objects  (.m .pos .vel .r)  - used only if dynamic = true
%   E       drift field     (make_plan_field)
%   dynamic true  -> the moving objects are predicted ahead (up to cfg.plan_H s)
%                    and avoided; if no path exists the extra margin is relaxed
%   path    N x 2 smoothed path [x y] in metres, 1 m spacing
%   info.relax  0..3 : how many times the moving-object margin was relaxed

    info.relax = 0;
    path = [];  ok = false;

    nTry = 1;
    if dynamic, nTry = 4; end

    found = false;  gp = [];
    for r = 0:nTry-1
        margin = cfg.dyn_margin*(1 - r/3);          % 2, 1.33, 0.67, 0 m
        [gp,found] = astar_grid(P,cfg,start,W,E,dynamic,margin);
        if found, info.relax = r; break; end
    end
    if ~found, return; end

    path = smooth_path(P,cfg,gp,W,E,dynamic);
    ok = true;
end

%% ------------------------------------------------------------------
function [gpath,found] = astar_grid(P,cfg,start,W,E,dynamic,margin)

    xs = P.xs;  n = numel(xs);  st = cfg.grid_step;
    ix0 = round((start(1)-xs(1))/st)+1;   iy0 = round((start(2)-xs(1))/st)+1;
    itx = round((cfg.target(1)-xs(1))/st)+1;  ity = round((cfg.target(2)-xs(1))/st)+1;

    blk = P.blk;
    blk(iy0,ix0) = false;  blk(ity,itx) = false;

    vmax = max(hypot(E.du(:),E.dv(:)));
    hf   = @(ix,iy) hypot(xs(ix)-cfg.target(1),xs(iy)-cfg.target(2))/(E.u+vmax);

    nb = [-1 -1; 0 -1; 1 -1; -1 0; 1 0; -1 1; 0 1; 1 1];

    gCost = inf(n,n);  fOpen = inf(n,n);  closed = false(n,n);
    parX  = zeros(n,n);  parY = zeros(n,n);  tArr = zeros(n,n);

    gCost(iy0,ix0) = 0;
    fOpen(iy0,ix0) = hf(ix0,iy0);
    found = false;

    while true
        [fb,lin] = min(fOpen(:));
        if isinf(fb), break; end
        [cy,cx] = ind2sub(size(fOpen),lin);
        if cx == itx && cy == ity, found = true; break; end
        fOpen(cy,cx) = inf;  closed(cy,cx) = true;

        for q = 1:size(nb,1)
            dx = nb(q,1);  dy = nb(q,2);
            nx = cx+dx;    ny = cy+dy;
            if nx < 1 || nx > n || ny < 1 || ny > n, continue; end
            if blk(ny,nx) || closed(ny,nx), continue; end
            if dx ~= 0 && dy ~= 0                       % no corner cutting
                if blk(cy,cx+dx) || blk(cy+dy,cx), continue; end
            end

            len = hypot(dx,dy);
            Lm  = len*st;
            gs  = max(E.u + (E.du(ny,nx)*dx + E.dv(ny,nx)*dy)/len, 0.3);
            tn  = tArr(cy,cx) + Lm/gs;

            if dynamic && W.m > 0                        % predicted moving objects
                tt = min(tn,cfg.plan_H);
                pa = W.pos + W.vel*tt;
                dd = hypot(pa(:,1)-xs(nx),pa(:,2)-xs(ny)) - W.r - cfg.boat_r - margin;
                if any(dd < 0), continue; end
            end

            newG = gCost(cy,cx) + Lm/gs + cfg.w_clear*P.pen(ny,nx)*Lm;
            if newG < gCost(ny,nx)
                gCost(ny,nx) = newG;
                parX(ny,nx) = cx;  parY(ny,nx) = cy;
                tArr(ny,nx) = tn;
                fOpen(ny,nx) = newG + hf(nx,ny);
            end
        end
    end

    gpath = [];
    if ~found, return; end

    cx = itx;  cy = ity;
    idx = [cx cy];
    while ~(cx == ix0 && cy == iy0)
        px = parX(cy,cx);  py = parY(cy,cx);
        if px == 0, found = false; gpath = []; return; end
        cx = px;  cy = py;
        idx = [cx cy; idx]; %#ok<AGROW>
    end
    gpath = [xs(idx(:,1))' xs(idx(:,2))'];
end

%% ------------------------------------------------------------------
function path = smooth_path(P,cfg,raw,W,E,dynamic)

    start = raw(1,:);  target = raw(end,:);

    % line-of-sight shortcuts (static plans only; dynamic plans keep the
    % time-checked nodes so that shortcuts cannot cut through a predicted ship)
    if dynamic
        wp = raw;
    else
        keep = 1;  i = 1;  nr = size(raw,1);
        while i < nr
            j = nr;
            while j > i+1 && ~los_free(P.safe_map,raw(i,:),raw(j,:))
                j = j-1;
            end
            keep(end+1) = j; %#ok<AGROW>
            i = j;
        end
        wp = raw(keep,:);
    end

    % resample every 1 m
    s  = [0; cumsum(hypot(diff(wp(:,1)),diff(wp(:,2))))];
    si = unique([(0:1:s(end))'; s(end)]);
    dense = [interp1(s,wp(:,1),si), interp1(s,wp(:,2),si)];

    % round the corners
    path = dense;
    for win = [11 9 7 5 3]
        cand = dense;
        cand(:,1) = movmean(dense(:,1),win);
        cand(:,2) = movmean(dense(:,2),win);
        cand(1,:) = start;  cand(end,:) = target;
        if acceptable(P,cfg,cand,W,E,dynamic)
            path = cand;
            break;
        end
    end
end

%% ------------------------------------------------------------------
function ok = acceptable(P,cfg,cand,W,E,dynamic)
    n  = size(P.safe_map,1);
    ix = min(n,max(1,round(cand(:,1))));
    iy = min(n,max(1,round(cand(:,2))));
    ok = all(P.safe_map(sub2ind(size(P.safe_map),iy,ix)) == 0);
    if ok && dynamic && W.m > 0
        sl = [0; cumsum(hypot(diff(cand(:,1)),diff(cand(:,2))))];
        tt = min(sl/E.u,cfg.plan_H);
        for a = 1:W.m
            px = W.pos(a,1)+W.vel(a,1)*tt;
            py = W.pos(a,2)+W.vel(a,2)*tt;
            if any(hypot(px-cand(:,1),py-cand(:,2)) <= W.r(a)+cfg.boat_r)
                ok = false;  return;
            end
        end
    end
end

%% ------------------------------------------------------------------
function ok = los_free(safe_map,p,q)
    n  = ceil(norm(q-p)/0.5)+1;
    xs = linspace(p(1),q(1),n);
    ys = linspace(p(2),q(2),n);
    idx = sub2ind(size(safe_map),round(ys),round(xs));
    ok = all(safe_map(idx) == 0);
end
