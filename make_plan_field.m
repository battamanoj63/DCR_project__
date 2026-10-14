function E = make_plan_field(P,cfg,sea,tk,ub) %#ok<INUSL>
% MAKE_PLAN_FIELD  Drift field (current + wind + wave drift) on the A* grid at time tk.
%   Same thing usv_run uses for (re)planning.  ub = commanded speed used by the planner.
    [xg,yg] = meshgrid(P.xs,P.xs);
    [cu,cv] = sea_current_ext(sea,xg,yg,tk);
    [wx,wy] = sea_wind(sea,tk);
    E.du   = cu+wx+sea.drift(1);
    E.dv   = cv+wy+sea.drift(2);
    E.u    = ub*max(0.5,1-sea.Kres*sea.Hs^2-0.1);
    E.dref = zeros(size(xg));
end
