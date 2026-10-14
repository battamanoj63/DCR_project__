function [eta,dEx,dEy] = sea_waves(sea,x,y,t)
% SEA_WAVES  Sea-surface elevation and slopes at (x,y,t).
%   eta     : elevation (m)
%   dEx,dEy : surface slope in world x and y
%   x and y may be scalars or equally sized arrays.

    wv  = sea.wave;
    eta = zeros(size(x));
    dEx = zeros(size(x));
    dEy = zeros(size(x));

    for i = 1:numel(wv.a)
        ph  = wv.k(i)*(x*cos(wv.dir(i)) + y*sin(wv.dir(i))) ...
              - wv.w(i)*t + wv.phi(i);
        eta = eta + wv.a(i)*cos(ph);
        sl  = -wv.a(i)*wv.k(i)*sin(ph);
        dEx = dEx + sl*cos(wv.dir(i));
        dEy = dEy + sl*sin(wv.dir(i));
    end
end
