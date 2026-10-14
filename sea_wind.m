function [wx,wy,wspd] = sea_wind(sea,t)
% SEA_WIND  Wind-induced drift velocity (slow variation + random gusts).
    idx  = min(numel(sea.gust),floor(t/sea.dt)+1);
    wspd = max(0, sea.wind_mean*(1+0.3*sin(0.04*t)) + sea.gust(idx));
    wdir = 25 + 15*sin(0.025*t);
    wx = wspd*cosd(wdir);
    wy = wspd*sind(wdir);
end
