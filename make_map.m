function [map,safe_map,rects] = make_map(safety_margin)
% MAKE_MAP  Water map used by every stage (same obstacles as your Stage 2).
%   map      : 1 = real obstacle, 0 = free water   (index as map(y,x))
%   safe_map : obstacles inflated by safety_margin cells
%   rects    : [xmin xmax ymin ymax] of each obstacle (for 3D drawing)

    if nargin < 1
        safety_margin = 4;
    end

    n = 100;
    map = zeros(n,n);

    map(35:50,20:35) = 1;     % Obstacle 1 - Vessel
    map(60:75,50:65) = 1;     % Obstacle 2 - Vessel
    map(20:30,65:80) = 1;     % Obstacle 3 - Restricted area
    map(70:82,15:25) = 1;     % Obstacle 4 - Rock / shallow water

    rects = [20 35 35 50;
             50 65 60 75;
             65 80 20 30;
             15 25 70 82];

    k = 2*safety_margin + 1;
    safe_map = double(conv2(map,ones(k),'same') > 0);
end
