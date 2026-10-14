function sea = make_sea(preset,T,dt,seed)
% MAKE_SEA  Builds the sea-state used by all stages.
%   preset : 'calm' | 'moderate' | 'rough'
%   T, dt  : simulation length and step (the wind-gust series is pre-computed)
%   seed   : random seed (own random stream, global rng is not touched)

    if nargin < 4, seed = 1; end

    %        wind  gust  current eddy   Hs    Tp
    switch lower(preset)
        case 'calm',     P = [0.04 0.01 0.05 0.05 0.2 3.0];
        case 'moderate', P = [0.10 0.03 0.10 0.12 0.6 4.0];
        case 'rough',    P = [0.20 0.08 0.18 0.25 1.2 5.0];
        otherwise,       error('Unknown sea preset: %s',preset);
    end

    rs = RandStream('mt19937ar','Seed',seed);

    sea.name       = [upper(preset(1)) lower(preset(2:end))];
    sea.wind_mean  = P(1);      % m/s  wind-induced drift speed
    sea.gust_sigma = P(2);      % m/s  gust intensity
    sea.cur_base   = P(3);      % m/s  background current
    sea.eddy_v     = P(4);      % m/s  peak eddy speed
    sea.Hs         = P(5);      % m    significant wave height
    sea.Tp         = P(6);      % s    peak wave period

    sea.eddy       = [60 45 15];% [x y radius] of the eddy
    sea.wave_dir   = 120;       % deg  direction the swell travels TOWARD
    sea.Kyaw       = 1.5;       % wave slope -> yaw disturbance (rad/s per slope)
    sea.Kres       = 0.06;      % added resistance: speed loss ~ Kres*Hs^2
    sea.Kdrift     = 0.02;      % wave drift speed ~ Kdrift*Hs^2 (m/s)
    sea.drift      = sea.Kdrift*sea.Hs^2*[cosd(sea.wave_dir) sind(sea.wave_dir)];

    % ---- irregular waves (Bretschneider spectrum, M components) ----
    M  = 20;
    g  = 9.81;
    wp = 2*pi/sea.Tp;
    w  = linspace(0.5*wp,3.5*wp,M);
    dw = w(2)-w(1);
    S  = (5/16)*sea.Hs^2*wp^4 ./ w.^5 .* exp(-1.25*(wp./w).^4);

    sea.wave.a   = sqrt(2*S*dw);
    sea.wave.w   = w;
    sea.wave.k   = w.^2/g;                                  % deep water
    sea.wave.phi = 2*pi*rand(rs,1,M);
    sea.wave.dir = deg2rad(sea.wave_dir) + deg2rad(15)*randn(rs,1,M);

    % ---- wind gusts: Ornstein-Uhlenbeck process, pre-computed ----
    n   = ceil(T/dt)+2;
    tau = 8;
    gust = zeros(1,n);
    for i = 2:n
        gust(i) = gust(i-1) - gust(i-1)/tau*dt + ...
                  sea.gust_sigma*sqrt(2*dt/tau)*randn(rs);
    end
    sea.gust = gust;
    sea.dt   = dt;
end
