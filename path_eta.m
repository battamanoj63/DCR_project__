function [eta,len] = path_eta(P,E,path)
% PATH_ETA  Estimated sailing time and length of a path, using the ground speed
%   (boat speed + drift component along the path) in the drift field E.
    d   = diff(path);
    L   = max(hypot(d(:,1),d(:,2)),1e-9);
    mid = 0.5*(path(1:end-1,:)+path(2:end,:));
    mx  = min(max(mid(:,1),P.xs(1)),P.xs(end));
    my  = min(max(mid(:,2),P.xs(1)),P.xs(end));
    du  = interp2(P.xs,P.xs,E.du,mx,my,'linear');
    dv  = interp2(P.xs,P.xs,E.dv,mx,my,'linear');
    gs  = max(E.u + (du.*d(:,1)+dv.*d(:,2))./L, 0.3);
    eta = sum(L./gs);
    len = sum(L);
end
