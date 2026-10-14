function [f,names] = risk_features(env,sea,x,y,psi,t,u)
% RISK_FEATURES  Feature vector (1 x 20) used by the ML collision-risk model.
%   Built from: geometry (clearance, bearing to nearest obstacle), the
%   mathematical wind / current / wave model, and the resulting ground
%   velocity of the vessel.  Scalars only.
%   (An onboard current/wind estimator is assumed; here the model values
%    are used directly.)

    ix = min(env.n,max(1,round(x)));
    iy = min(env.n,max(1,round(y)));

    d  = env.clear_d(iy,ix);
    ox = env.near_x(iy,ix) - x;
    oy = env.near_y(iy,ix) - y;
    dn = max(hypot(ox,oy),1e-6);
    ux = ox/dn;
    uy = oy/dn;
    rel = psi - atan2(oy,ox);          % 0 = heading straight at the obstacle

    [cx,cy]     = sea_current(sea,x,y,t);
    [wx,wy]     = sea_wind(sea,t);
    [~,dEx,dEy] = sea_waves(sea,x,y,t);

    long_s =  dEx*cos(psi) + dEy*sin(psi);
    lat_s  = -dEx*sin(psi) + dEy*cos(psi);
    u_eff  = u*max(0.5,1 - sea.Kres*sea.Hs^2 - 0.5*abs(atan(long_s)));

    ex = cx + wx + sea.drift(1);       % environmental drift
    ey = cy + wy + sea.drift(2);
    vx = u_eff*cos(psi) + ex;          % ground velocity
    vy = u_eff*sin(psi) + ey;

    closing = vx*ux + vy*uy;           % > 0 : moving towards the obstacle
    d_along =  ex*cos(psi) + ey*sin(psi);
    d_cross = -ex*sin(psi) + ey*cos(psi);

    f = [x/100, y/100, cos(psi), sin(psi), d, 1/(1+d), cos(rel), sin(rel), ...
         cx, cy, wx, wy, closing, closing/max(d,0.5), hypot(vx,vy), sea.Hs, ...
         long_s, lat_s, d_along, d_cross];

    if nargout > 1
        names = {'x','y','cos(psi)','sin(psi)','clearance','1/(1+clearance)', ...
                 'cos(rel.bearing)','sin(rel.bearing)','current_x','current_y', ...
                 'wind_x','wind_y','closing speed','closing/clearance', ...
                 'ground speed','Hs','pitch slope','roll slope', ...
                 'drift along','drift cross'};
    end
end