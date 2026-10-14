function draw_map(ax,P,show_safe)
% DRAW_MAP  Water + static obstacles (grey; the safety margin is drawn lighter).
    if nargin < 3, show_safe = true; end
    axes(ax); hold(ax,'on');                                  %#ok<LAXES>
    img = P.map + 0.35*show_safe*P.safe_map.*(1-P.map);
    imagesc(ax,[1 100],[1 100],img);
    set(ax,'YDir','normal');
    colormap(ax,[1 1 1; 0.88 0.88 0.88; 0.45 0.45 0.45]);
    caxis(ax,[0 1]);
    axis(ax,'equal'); axis(ax,[1 100 1 100]); grid(ax,'on'); box(ax,'on');
end
