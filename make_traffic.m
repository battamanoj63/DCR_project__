function tr = make_traffic(sea,cfg,t)
% MAKE_TRAFFIC  Moving objects of the ocean: two ships, floating debris
%   (drifts with current + wind + wave drift) and a patrol boat.
%   tr.m            number of objects
%   tr.r            m x 1 radius (m)
%   tr.pos(:,:,k)   m x 2 position at time t(k)
%   tr.vel(:,:,k)   m x 2 velocity at time t(k)

    dt = t(2)-t(1);
    Nt = numel(t);

    %        x0   y0    vx    vy    r
    A = [   80   72   0.35  0.00  3.0;     % ship 1 (crosses the east route)
            50   88   0.40  0.00  3.5;     % ship 2 (crosses the north route)
            60   42   0.00  0.00  1.5;     % floating debris (velocity from the sea)
            30   60   0.50  0.25  2.0 ];   % patrol boat
    isDebris = [false false true false];

    m   = size(A,1);
    pos = zeros(m,2,Nt);
    vel = zeros(m,2,Nt);
    pos(:,:,1) = A(:,1:2);
    v0 = A(:,3:4);

    lo = 3;  hi = 97;
    for k = 1:Nt
        for a = 1:m
            if isDebris(a)
                [cx,cy] = sea_current_ext(sea,pos(a,1,k),pos(a,2,k),t(k));
                [wx,wy] = sea_wind(sea,t(k));
                v = [cx+wx+sea.drift(1), cy+wy+sea.drift(2)];
            else
                v = v0(a,:);
            end
            vel(a,:,k) = v;
        end
        if k < Nt
            pn = pos(:,:,k) + vel(:,:,k)*dt;
            for a = 1:m                        % bounce at the map border
                for c = 1:2
                    if pn(a,c) < lo || pn(a,c) > hi
                        pn(a,c) = min(hi,max(lo,pn(a,c)));
                        if ~isDebris(a), v0(a,c) = -v0(a,c); end
                    end
                end
            end
            pos(:,:,k+1) = pn;
        end
    end

    tr.m   = m;
    tr.r   = A(:,5);
    tr.pos = pos;
    tr.vel = vel;
    tr.names = {'Ship 1','Ship 2','Debris','Patrol boat'};
end
