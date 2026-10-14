function [cx,cy] = sea_current(sea,x,y,t)
% SEA_CURRENT  Water current at position (x,y) and time t.
%   = tidal pulsation + shear across the map + a circular eddy.
%   x and y may be scalars or equally sized arrays.

    spd = sea.cur_base*(1+0.3*sin(0.03*t+1)).*(1+0.4*(y-50)/50);
    dir = 140 + 10*cos(0.02*t);
    cx  = spd.*cosd(dir);
    cy  = spd.*sind(dir);

    % Rankine vortex
    dx = x - sea.eddy(1);
    dy = y - sea.eddy(2);
    R  = sea.eddy(3);
    r  = max(hypot(dx,dy),1e-6);
    vt = sea.eddy_v*min(r/R, R./r);

    cx = cx - vt.*dy./r;
    cy = cy + vt.*dx./r;
end
