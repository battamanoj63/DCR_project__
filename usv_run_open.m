function res = usv_run_open(cfg,P,sea,tr,start,path,noise_seed)
% USV_RUN_OPEN  Mission of one boat WITHOUT any PID / feedback controller.
%   The boat dead-reckons along the planned path: a pre-computed (feed-forward)
%   turn-rate schedule follows the heading of the plan at the nominal speed.
%   Nothing measures the position error, so nothing corrects it:
%     - current, wind and wave drift push the boat off the plan,
%     - the wave yaw disturbance (Kyaw*slope) turns the bow and is never cancelled,
%     - no avoidance of moving objects and no replanning.
%   This is the "what does the rough sea do to the journey" baseline.
%   The vehicle model, sea model and metrics are identical to usv_run.m.
    dt = cfg.dt;  N = floor(cfg.T/dt)+1;  t = (0:N-1)*dt;
    target = cfg.target;
    s0 = rng;  rng(noise_seed);

    % ---- reference (arc-length) description of the plan
    seg = hypot(diff(path(:,1)),diff(path(:,2)));
    s   = [0; cumsum(seg)];
    ok  = [true; seg > 1e-9];
    path = path(ok,:);  s = s(ok);
    Ltot = s(end);
    hdg  = atan2(diff(path(:,2)),diff(path(:,1)));
    hdg  = unwrap([hdg; hdg(end)]);
    href = @(sq) interp1(s,hdg,min(max(sq,0),Ltot),'linear','extrap');

    eta_nom = Ltot/cfg.u_nom;                       % planned sailing time
    Tend    = min(cfg.T, 1.8*eta_nom+30);           % give the boat extra time to arrive
    Nend    = min(N,floor(Tend/dt)+1);

    x = zeros(1,N); y = x; psi = x;  xte_l = zeros(1,N);
    x(1) = start(1); y(1) = start(2); psi(1) = href(0);
    pitch_s = 0;
    coll = 0; in_coll = false(tr.m,1); static_hits = 0; in_static = false;
    min_gap = inf; min_clear = inf;
    closest = inf; miss_eta = nan; reached = false; last = Nend;
    k_eta = round(eta_nom/dt)+1;

    for k = 1:Nend-1
        pos = [x(k) y(k)];  tk = t(k);
        dT  = norm(target-pos);
        closest = min(closest,dT);
        if k == k_eta, miss_eta = dT; end
        if dT <= cfg.thresh
            reached = true; last = k;
            x(k+1:end) = x(k); y(k+1:end) = y(k); psi(k+1:end) = psi(k);
            break;
        end

        [wx,wy] = sea_wind(sea,tk);
        [cx,cy] = sea_current_ext(sea,x(k),y(k),tk);
        [~,dEx,dEy] = sea_waves(sea,x(k),y(k),tk);
        dvec   = [wx+cx+sea.drift(1), wy+cy+sea.drift(2)];
        long_s =  dEx*cos(psi(k)) + dEy*sin(psi(k));
        lat_s  = -dEx*sin(psi(k)) + dEy*cos(psi(k));
        pitch_s = pitch_s + dt*(atan(long_s)-pitch_s)/0.6;

        % feed-forward turn rate from the schedule  s_ref = u_nom * t
        r_ff  = wrap(href(cfg.u_nom*t(k+1))-href(cfg.u_nom*t(k)))/dt;
        u_eff = cfg.u_nom*max(0.5,1-sea.Kres*sea.Hs^2-0.5*abs(pitch_s));
        psi(k+1) = psi(k) + (r_ff + sea.Kyaw*lat_s)*dt;
        x(k+1) = min(98,max(2,x(k) + (u_eff*cos(psi(k))+dvec(1))*dt));
        y(k+1) = min(98,max(2,y(k) + (u_eff*sin(psi(k))+dvec(2))*dt));

        xte_l(k) = min(hypot(path(:,1)-x(k),path(:,2)-y(k)));

        pn = tr.pos(:,:,k+1);
        gp = hypot(pn(:,1)-x(k+1),pn(:,2)-y(k+1)) - tr.r - cfg.boat_r;
        min_gap = min(min_gap,min(gp));
        newhit  = (gp <= 0) & ~in_coll;
        coll    = coll + sum(newhit);
        in_coll = (gp <= 0) | (in_coll & gp < 1);
        iy = round(y(k+1)); ix = round(x(k+1));
        hit = P.map(iy,ix) == 1;
        if hit && ~in_static, static_hits = static_hits+1; end
        in_static = hit;
        min_clear = min(min_clear,P.clear_d(iy,ix));
    end
    if ~reached, last = Nend; end
    if isnan(miss_eta), miss_eta = norm(target-[x(last) y(last)]); end
    xte_l(last) = xte_l(max(1,last-1));

    kk  = 1:last;
    sgm = hypot(diff(x(kk)),diff(y(kk)));
    res.reached     = reached;
    res.t_end       = t(last);
    res.final_err   = norm(target-[x(last) y(last)]);
    res.closest     = closest;                         % closest approach to the incident
    res.miss_eta    = miss_eta;                        % miss distance when the plan said "arrive"
    res.eta_plan    = eta_nom;
    res.path_len    = sum(sgm);
    res.straight    = norm(target-start);
    res.detour      = res.path_len/res.straight;
    res.mean_speed  = res.path_len/max(t(last),dt);
    res.collisions  = coll;
    res.static_hits = static_hits;
    res.success     = reached && coll == 0 && static_hits == 0;
    res.min_gap     = min_gap;
    res.min_clear   = min_clear;
    res.xte_rmse    = sqrt(mean(xte_l(kk).^2));
    res.xte_max     = max(xte_l(kk));
    res.t = t(kk); res.x = x(kk); res.y = y(kk); res.psi = psi(kk); res.xte = xte_l(kk);
    res.path = path;
    rng(s0);
end

function a = wrap(a), a = mod(a+pi,2*pi)-pi; end
