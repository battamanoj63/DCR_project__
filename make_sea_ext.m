function sea = make_sea_ext(preset,T,dt,seed)
% MAKE_SEA_EXT  Same sea model as make_sea.m, with two extra sea states
%   ('heavy','severe') and additional eddies.
%   sea.eddy   = [x y R]      main eddy (as in make_sea, used by sea_current)
%   sea.eddies = [x y R v]    EXTRA eddies (rows), used by sea_current_ext

    if nargin < 4, seed = 1; end

    %        wind  gust  current eddy   Hs    Tp
    switch lower(preset)
        case 'calm',     Pm = [0.04 0.01 0.05 0.05 0.2 3.0];  nEx = 0;
        case 'moderate', Pm = [0.10 0.03 0.10 0.12 0.6 4.0];  nEx = 1;
        case 'rough',    Pm = [0.20 0.08 0.18 0.25 1.2 5.0];  nEx = 1;
        case 'heavy',    Pm = [0.28 0.11 0.24 0.33 1.7 6.0];  nEx = 2;
        case 'severe',   Pm = [0.35 0.14 0.28 0.40 2.2 6.5];  nEx = 2;
        otherwise,       error('Unknown sea preset: %s',preset);
    end

    rs = RandStream('mt19937ar','Seed',seed);

    sea.name       = [upper(preset(1)) lower(preset(2:end))];
    sea.wind_mean  = Pm(1);
    sea.gust_sigma = Pm(2);
    sea.cur_base   = Pm(3);
    sea.eddy_v     = Pm(4);
    sea.Hs         = Pm(5);
    sea.Tp         = Pm(6);

    sea.eddy       = [60 45 15];
    sea.wave_dir   = 120;
    sea.Kyaw       = 1.5;
    sea.Kres       = 0.06;
    sea.Kdrift     = 0.02;
    sea.drift      = sea.Kdrift*sea.Hs^2*[cosd(sea.wave_dir) sind(sea.wave_dir)];

    % ---- extra eddies [x y R v]  (opposite rotation sense for the 2nd one)
    ex = [30 20 12  0.8*sea.eddy_v;
          78 72 10 -0.8*sea.eddy_v];
    sea.eddies = ex(1:nEx,:);

    % ---- irregular waves (Bretschneider spectrum, M components)
    M  = 20;
    g  = 9.81;
    wp = 2*pi/sea.Tp;
    w  = linspace(0.5*wp,3.5*wp,M);
    dw = w(2)-w(1);
    S  = (5/16)*sea.Hs^2*wp^4 ./ w.^5 .* exp(-1.25*(wp./w).^4);

    sea.wave.a   = sqrt(2*S*dw);
    sea.wave.w   = w;
    sea.wave.k   = w.^2/g;
    sea.wave.phi = 2*pi*rand(rs,1,M);
    sea.wave.dir = deg2rad(sea.wave_dir) + deg2rad(15)*randn(rs,1,M);

    % ---- wind gusts: Ornstein-Uhlenbeck process
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
