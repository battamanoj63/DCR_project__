function [cx,cy] = sea_current_ext(sea,x,y,t)
% SEA_CURRENT_EXT  sea_current (tide + shear + main eddy) plus the extra
%   Rankine eddies in sea.eddies = [x y R v].
%   x and y may be scalars or equally sized arrays.

    [cx,cy] = sea_current(sea,x,y,t);

    if isfield(sea,'eddies')
        for e = 1:size(sea.eddies,1)
            dx = x - sea.eddies(e,1);
            dy = y - sea.eddies(e,2);
            R  = sea.eddies(e,3);
            r  = max(hypot(dx,dy),1e-6);
            vt = sea.eddies(e,4)*min(r/R, R./r);
            cx = cx - vt.*dy./r;
            cy = cy + vt.*dx./r;
        end
    end
end
