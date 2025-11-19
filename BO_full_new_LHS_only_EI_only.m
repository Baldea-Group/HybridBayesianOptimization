function [hist_best, iter_y, n_iter, best_proc, best_cat, best_y] = ...
         BO_full_new_LHS_only_EI_only(doPlot)
% Full 5D Bayesian Optimization (scaled) for A->B->C consecutive reactor.
% Now uses MATLAB's fitrgp for the GP surrogate.
%
% Outputs:
%   hist_best : [n_iter x 1] best objective (100*Y_B) after each iteration
%   iter_y    : [n_iter x 1] objective at each iteration (100*Y_B)
%   n_iter    : scalar number of BO iterations
%   best_proc : [T, P, tau] at final best
%   best_cat  : [Eb, dEa] at final best
%   best_y    : scalar best objective (100*Y_B)

if nargin < 1 || isempty(doPlot)
    doPlot = true;    % default: make plots + CSV, as before
end

%rng(42);
%rng(25966500);
set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% ---------- Bounds (real units) ----------
lb = [ 450,   1,   0.1,  -2.0, -0.2];   % [T(K), P(bar), tau(s), Eb(eV), dEa(eV)]
ub = [ 700,  20,   5.0,   0.0,  0.2];

% --- bounds ---
d  = numel(lb);
ltau_lb = log(lb(3)); 
ltau_ub = log(ub(3));

toUnit = @(X) [ ...
    (X(:,1)-lb(1)) ./ (ub(1)-lb(1)+eps), ...                       % T
    (X(:,2)-lb(2)) ./ (ub(2)-lb(2)+eps), ...                       % P
    (log(X(:,3)) - ltau_lb) ./ (ltau_ub - ltau_lb + eps), ...      % tau (log)
    (X(:,4)-lb(4)) ./ (ub(4)-lb(4)+eps), ...                       % Eb
    (X(:,5)-lb(5)) ./ (ub(5)-lb(5)+eps)];                          % dEa

fromUnit = @(Z) [ ...
    lb(1) + Z(:,1).*(ub(1)-lb(1)), ...
    lb(2) + Z(:,2).*(ub(2)-lb(2)), ...
    exp(ltau_lb + Z(:,3).*(ltau_ub - ltau_lb)), ...
    lb(4) + Z(:,4).*(ub(4)-lb(4)), ...
    lb(5) + Z(:,5).*(ub(5)-lb(5))];

% ---------- BO knobs ----------
n_init = 36;
n_iter = 100;
acq_M  = 8000;          % candidate pool size
costlywaittime = 0.0;   % emulate black box delay (sec)
% highT_frac = 0.40;      % ~40% of candidates biased to high T
% 
% % dimensions
% T_dim = 1;              % index of T in [T, P, tau, Eb, dEa]
% T_high_min = 0.80;      % top 20% of T (unit space)

% ---------- Initial design: pure LHS in unit space ----------
U0 = lhs_unit(n_init, d);      % n_init × d in [0,1]^d
X  = fromUnit(U0);             % map to real units

% Evaluate initial design
y  = zeros(size(X,1),1);
for i = 1:size(X,1)
    y(i) = det_objective_consecutive(X(i,:), costlywaittime);
end

best_y = max(y);
[~, ib] = max(y);
best_x = X(ib,:);

hist_best = zeros(n_iter,1);
iter_y    = zeros(n_iter,1);

fprintf('---- Initial evaluations (n=%d) ----\n', size(X,1));
for i = 1:size(X,1)
    fprintf('init %3d | f=%8.4f  [T=%7.2f, P=%6.2f, tau=%5.2f, Eb=%6.3f, dEa=%6.3f]\n', ...
        i, y(i), X(i,1), X(i,2), X(i,3), X(i,4), X(i,5));
end

tstart = tic;

% ---------- BO loop ----------
for t = 1:n_iter
    % --- Train GP surrogate using MATLAB's fitrgp on unit-space inputs ---
    Utrain = toUnit(X);
    gprMdl = fitrgp(Utrain, y, ...
        'KernelFunction','ardsquaredexponential', ...
        'Standardize',true);

    % Extract kernel hyperparameters (length-scales + signal variance)
    params = gprMdl.KernelInformation.KernelParameters;
    ell = params(1:d);
    sf  = params(end);
    sn_eff = gprMdl.Sigma;

    fprintf('Iter %2d | ell=[%s], sf=%.3g, sn_eff=%.3g | std(y)=%.3g\n', ...
        t, sprintf('%.3g ', ell), sf, sn_eff, std(y));

    % % ---- Candidate pool (unit) ----
    % M_hiT = round(highT_frac*acq_M);
    % U_hiT = rand(M_hiT, d);
    % U_hiT(:,T_dim) = T_high_min + (1 - T_high_min)*rand(M_hiT,1);
    % 
    % U_lhs_cand = lhs_unit(round(0.25*acq_M), d);
    % U_rnd      = rand(round(0.25*acq_M), d);
    % best_u     = toUnit(best_x);
    % U_jit      = min(max(best_u + 0.06*randn(round(0.25*acq_M), d), 0), 1);
    % E          = edge_sweeps(d, 25);
    % 
    % Ucand = [U_hiT; U_lhs_cand; U_rnd; U_jit; E];   % no feasibility filter



        % ---- Candidate pool (unit, neutral – no T bias) ----
    Nc          = acq_M;
    n_lhs_cand  = round(0.4 * Nc);
    n_rnd       = round(0.3 * Nc);
    n_jit       = round(0.3 * Nc);

    U_lhs_cand  = lhs_unit(n_lhs_cand, d);         % LHS over [0,1]^d
    U_rnd       = rand(n_rnd, d);                  % uniform random
    best_u      = toUnit(best_x);                  % current best in unit space
    U_jit       = min(max(best_u + 0.06*randn(n_jit, d), 0), 1);   % jitter around best
    E           = edge_sweeps(d, 25);              % edge sweeps of the hypercube

    Ucand       = [U_lhs_cand; U_rnd; U_jit; E];   % no feasibility filter


    % ---- Predict & EI ----
    [mu_c, s2_c] = predict(gprMdl, Ucand);
    s_c = sqrt(max(s2_c, 0));

    % Moderately exploratory xi schedule
    % xi0 = 0.5; 
    % xi  = max(0.10, xi0 * (0.98^(t-1)));
    xi = 0; 
    EI  = expected_improvement_with_xi(mu_c, s_c, best_y, xi);

    % ---- Argmax EI ----
    [~, idx] = max(EI);
    x_next = fromUnit(Ucand(idx,:));
    mu_at  = mu_c(idx); 
    s_at   = s_c(idx);

    % Evaluate objective
    y_next = det_objective_consecutive(x_next, costlywaittime);

    % Update data
    X = [X; x_next]; 
    y = [y; y_next]; 

    if y_next > best_y
        best_y = y_next; best_x = x_next;
    end
    hist_best(t) = best_y;
    iter_y(t)    = y_next;

    fprintf('  -> pick f=%8.4f (mu=%8.4f, s=%6.4f)  [T=%7.2f, P=%6.2f, tau=%5.2f, Eb=%6.3f, dEa=%6.3f]\n', ...
        y_next, mu_at, s_at, x_next(1), x_next(2), x_next(3), x_next(4), x_next(5));
end

telapsed = toc(tstart);

% ---------- Results ----------
fprintf('\n=== 5D BO (consecutive; no Kads penalty) complete ===\n');
fprintf('Time elapsed = %.3f s\n', telapsed);
fprintf('Best f = %.6f\n', best_y);
fprintf('Best x*:  T=%.3f K,  P=%.3f bar,  tau=%.3f s,  Eb=%.4f eV,  dEa=%.4f eV\n', ...
    best_x(1), best_x(2), best_x(3), best_x(4), best_x(5));

% Split best_x into process and catalyst parts for output
best_proc = best_x(1:3);   % [T, P, tau]
best_cat  = best_x(4:5);   % [Eb, dEa]

% Refit final GP (for plots) using all data
Utrain = toUnit(X);
gprMdl = fitrgp(Utrain, y, ...
    'KernelFunction','ardsquaredexponential', ...
    'Standardize',true);

if doPlot
    % ---------- Progress plots ----------
    figure('Name','BO progress','Color','w');
    subplot(2,1,1);
    plot(1:n_iter, hist_best, 'k-', 'LineWidth',1.5); grid on;
    xlabel('BO Iteration'); ylabel('Best $Y_B$'); title('Best-so-far');
    subplot(2,1,2);
    stem(1:n_iter, iter_y, 'filled'); grid on;
    xlabel('BO Iteration'); ylabel('$Y_B$ at iteration'); title('Per-iteration objective');
end

% ensure an 'exports' folder exists
outDir = 'exports';
if ~exist(outDir,'dir'); mkdir(outDir); end

% Timestamped filename (progress CSV)
outFile = fullfile(outDir, sprintf('Full_BO_progress_%s.csv', ...
                                   datestr(now,'yyyymmdd_HHMMSS')));
% (You can fill this if you want to export progress vectors.)

if doPlot
    % ---------- 1D μ±2σ & EI (mid slice) ----------
    labels = {'$T$ (K)','$P$ (bar)','$\tau$ (s)','$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'};
    figure('Name','1D GP profiles (slice through mid)','Color','w');
    mid = 0.5*(lb + ub);
    f_best = max(y);
    for k = 1:d
        xx = linspace(lb(k), ub(k), 250)';           
        Xs = repmat(mid, numel(xx), 1);  Xs(:,k) = xx;
        [mu, s2] = predict(gprMdl, toUnit(Xs));
        s = sqrt(max(s2,0));
        EIplot = expected_improvement_with_xi(mu, s, f_best, 0.10);

        subplot(2,5,k); hold on; grid on;
        fill([xx; flipud(xx)], [mu-2*s; flipud(mu+2*s)], [0.88 0.88 0.88], 'EdgeColor','none');
        plot(xx, mu, 'k-', 'LineWidth',1.2);
        ylabel('$\mu \pm 2\sigma$'); xlabel(labels{k});
        title(sprintf('GP mean: %s', labels{k}));
        subplot(2,5,5+k);
        plot(xx, EIplot, 'k-', 'LineWidth',1.2); grid on;
        ylabel('EI'); xlabel(labels{k});
        title(sprintf('EI: %s', labels{k}));
    end
end

if doPlot
    % ---------- 1D μ±2σ & EI at best_x ----------
    figure('Name','Surrogates (top) and Expected Improvement (bottom) @ x*','Color','w');
    ref = best_x;  f_best = max(y);
    labels = {'$T$ (K)','$P$ (bar)','$\tau$ (s)','$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'};
    for k = 1:d
        xx = linspace(lb(k), ub(k), 250)';     
        Xs = repmat(ref, numel(xx), 1);  Xs(:,k) = xx;
        [mu, s2] = predict(gprMdl, toUnit(Xs));
        s = sqrt(max(s2,0));
        EIplot = expected_improvement_with_xi(mu, s, f_best, 0.10);

        subplot(2,5, k);  hold on; grid on;
        fill([xx; flipud(xx)], [mu-2*s; flipud(mu+2*s)], [0.88 0.88 0.88], 'EdgeColor','none');
        plot(xx, mu, 'k-', 'LineWidth',1.25);
        xlabel(labels{k}); ylabel('$GP \mu \pm 2\sigma$');

        subplot(2,5, 5 + k);
        plot(xx, EIplot, 'k-', 'LineWidth',1.25); grid on;
        xlabel(labels{k}); ylabel('EI');
        title(sprintf('EI: %s', labels{k}));
    end
end

if doPlot
    % ---------- 1D μ±2σ (top) & EI (bottom) at best_x in tiled layout ----------
    figure('Name','Surrogates (top) and Expected Improvement (bottom) @ x* (tiled)','Color','w');

    ref    = best_x;                 
    f_best = max(y);                 
    ncols  = 2;                      
    nrows_tiles = 6;                 
    tl = tiledlayout(nrows_tiles, ncols, 'TileSpacing','compact','Padding','compact');

    tile = @(r,c) (r-1)*ncols + c;

    labels = {'$T$ (K)','$P$ (bar)','$\tau$ (s)','$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'};
    max_show = min(d,5);

    for k = 1:max_show
        if k <= 2
            base_row = 1; col = k;        
        elseif k <= 4
            base_row = 3; col = k-2;      
        else
            base_row = 5; col = 1;        
        end

        xx = linspace(lb(k), ub(k), 250)';     
        Xs = repmat(ref, numel(xx), 1);  
        Xs(:,k) = xx;

        [mu, s2] = predict(gprMdl, toUnit(Xs));
        s = sqrt(max(s2,0));
        EIplot = expected_improvement_with_xi(mu, s, f_best, 0.10);

        % ---- Top tile: μ ± 2σ ----
        ax1 = nexttile(tile(base_row, col)); 
        hold(ax1, 'on'); grid(ax1, 'on');
        fill(ax1, [xx; flipud(xx)], [mu-2*s; flipud(mu+2*s)], [0.88 0.88 0.88], 'EdgeColor','none');
        plot(ax1, xx, mu, 'k-', 'LineWidth', 1.25);
        xlabel(ax1, labels{k}); ylabel(ax1, '$GP \mu \pm 2\sigma$', 'Interpreter','latex');

        % ---- Bottom tile: EI ----
        ax2 = nexttile(tile(base_row+1, col)); 
        plot(ax2, xx, EIplot, 'k-', 'LineWidth', 1.25); grid(ax2, 'on');
        xlabel(ax2, labels{k}); ylabel(ax2, 'EI', 'Interpreter','latex');
    end

    set(findall(gcf,'-property','FontName'),'FontName','Calibri')
    set(findall(gcf,'-property','FontSize'),'FontSize',12)
end

if doPlot
    % ---------- GP μ±2σ (left y-axis) + EI (right y-axis), tiled ----------
    tile_w_in = 3;       % inches
    tile_h_in = 2;       % inches

    nCols = min(2, max(1,d));
    nRows = ceil(d / nCols);

    fig_w_in = nCols * tile_w_in;
    fig_h_in = nRows * tile_h_in;

    figure('Name','GP mean ± 2σ (left) & EI (right) @ x*', ...
           'Color','w', ...
           'Units','inches', ...
           'Position',[1 1 fig_w_in fig_h_in]);

    t = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');

    ref     = best_x;       
    x_best  = best_x;
    f_best  = max(y);      

    labels = {'$T$ (K)','$P$ (bar)','$\tau$ (s)','$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'};
    series = cell(d,1);

    for k = 1:d
        xx = linspace(lb(k), ub(k), 250)';          
        Xs = repmat(ref, numel(xx), 1);
        Xs(:,k) = xx;

        [mu, s2] = predict(gprMdl, toUnit(Xs));
        s = sqrt(max(s2,0));

        xi = 0.10;
        EIplot = expected_improvement_with_xi(mu, s, f_best, xi);

        ax = nexttile; hold(ax,'on'); grid(ax,'on');

        yyaxis(ax,'left');
        fill(ax, [xx; flipud(xx)], [mu - 2*s; flipud(mu + 2*s)], ...
             [0.88 0.88 0.88], 'EdgeColor','none');
        plot(ax, xx, mu, 'k-', 'LineWidth', 1.25);
        ylabel(ax, '$GP\ \mu \pm 2\sigma$', 'Interpreter','latex');

        if exist('best_f','var') && ~isempty(best_f)
            yline(ax, best_f, '--');
        end
        if exist('x_best','var') && numel(x_best)>=k
            xline(ax, x_best(k), ':','LineWidth',2);
        end

        yyaxis(ax,'right');
        plot(ax, xx, EIplot, '--', 'LineWidth', 1.25);
        ylabel(ax, 'EI', 'Interpreter','latex');

        xlabel(ax, labels{k});
        dx = ub(k) - lb(k);
        xlim(ax, [lb(k) - 0.05*dx, ub(k) + 0.05*dx]);
        box on; hold(ax,'off');

        series{k} = table( ...
            repmat(string(labels{k}), numel(xx), 1), ...
            repmat(k,               numel(xx), 1), ...
            xx, ...
            mu, ...
            s, ...
            mu - 2*s, ...
            mu + 2*s, ...
            EIplot, ...
            repmat(xi,             numel(xx), 1), ...
            'VariableNames', {'Variable','Subplot','X','Mu','Sigma','Mu_minus_2Sigma','Mu_plus_2Sigma','EI','xi'} );
    end

    outDir = 'exports';
    if ~exist(outDir,'dir'); mkdir(outDir); end
    exportData = vertcat(series{:});
    outFile = fullfile(outDir, sprintf('Full_BO_%s.csv', datestr(now, 'yyyymmdd_HHMMSS')));
    writetable(exportData, outFile);
    fprintf('CSV exported: %s\n', outFile);

    set(findall(gcf,'-property','FontName'),'FontName','Helvetica');
    set(findall(gcf,'-property','FontSize'),'FontSize',12);

    print_opt_solution_table_columns(best_x(1:3), best_x(4:5), best_y);

    set(findall(gcf,'-property','FontName'),'FontName','Calibri')
    set(findall(gcf,'-property','FontSize'),'FontSize',12)
end

end % ===== main =====

% ===================== Objective (A->B->C; NO penalty) =====================
function f = det_objective_consecutive(x, waittime)
T   = x(1); P = x(2); tau = x(3); Eb = x(4); dEa = x(5);

% constants & param
R = 8.314;   A1=1e7;  A2=3e6;  Pref=8.0;
beta1=1.0;   beta2=0.35;      % pressure orders
Eb0=-1.0;    sigmaE=0.40;     % volcano center/width
alpha=0.6;                    % reduces side rate near volcano peak
K0=0.06;     kappa=3.0;       % adsorption strength

[volc, Ea1, Ea2] = costly_black_box_simple(Eb, dEa, Eb0, sigmaE, waittime);
phi1 = exp(-Ea1/(R*T));
phi2 = exp(-Ea2/(R*T));
pf1 = exp(beta1*log(P/Pref));
pf2 = exp(beta2*log(P/Pref));
Kads = K0 * exp(kappa*(Eb - Eb0));
denL = (1 + Kads*P)^2;

k1 = A1 * volc              * phi1 * pf1 / denL;
k2 = A2 * (1 - alpha*volc)  * phi2 * pf2 / denL;

% Yield of B in A->B->C with residence time tau
Y = (k1*tau) / ((1 + k1*tau)*(1 + k2*tau));

f = 100*Y;   % pure yield; NO penalties
end

function [volc, Ea1, Ea2] = costly_black_box_simple(Eb, dEa, Eb0, sigmaE, waittime)
if nargin < 5, waittime = 0; end
volc = exp(-0.5*((Eb - Eb0)/sigmaE).^2);
Ea1  = 80000 - 20000*dEa;
Ea2  = 95000 +  5000*dEa;
if waittime > 0, pause(waittime); end
end

% ===================== Acquisition (EI) using standard MATLAB stats =====================
function EI = expected_improvement_with_xi(mu, s, f_best, xi)
%EXPECTED_IMPROVEMENT_WITH_XI  EI acquisition (maximization) with offset xi.
% Uses normcdf / normpdf from the Statistics and Machine Learning Toolbox.
    s = max(s, 1e-12);
    impr = mu - f_best - xi;
    z = impr ./ s;

    Phi = normcdf(z);
    phi = normpdf(z);

    EI = impr .* Phi + s .* phi;
    EI(s <= 1e-12) = 0;
    EI = max(EI, 0);
end

% ===================== Utilities =====================
function A = lhs_unit(n, d)
A = zeros(n, d);
for j = 1:d
    p = randperm(n)';                      
    A(:, j) = ((p - 1) + rand(n,1)) / n;   
end
end

function E = edge_sweeps(d, m)
E = [];
mid = 0.5*ones(1,d);
for k = 1:d
    Z = repmat(mid, m, 1);
    Z(:,k) = linspace(0,1,m)';
    E = [E; Z]; %#ok<AGROW>
end
end

% ===================== Table-printing helpers (unchanged) =====================
function print_opt_solution_table_columns(best_proc, best_cat, best_y)
T   = best_proc(1);
P   = best_proc(2);
tau = best_proc(3);
Eb  = best_cat(1);
dEa = best_cat(2);
Y   = best_y;

fprintf('\n');
fprintf('\\begin{table}[t]\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{lcccccc}\n');
fprintf('\\toprule\n');
fprintf('Scenario & $T$ [K] & $P$ [bar] & $\\tau$ [s] & $E_b$ [eV] & $\\Delta E_a$ [eV] & Yield $Y$ \\\\\n');
fprintf('\\midrule\n');
fprintf('& %.2f & %.2f & %.3f & %.3f & %.3f & %.4f \\\\\n', T, P, tau, Eb, dEa, Y);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\caption{Optimal operating point}\n');
fprintf('\\label{tab:optimum}\n');
fprintf('\\end{table}\n');
fprintf('\n');
end

function print_opt_solution_table(best_proc, best_cat, best_y)
T   = best_proc(1);
P   = best_proc(2);
tau = best_proc(3);
Eb  = best_cat(1);
dEa = best_cat(2);
Y   = best_y;

fprintf('\n');
fprintf('\\begin{table}[t]\n');
fprintf('\\centering\n');
fprintf('\\begin{tabular}{l r}\n');
fprintf('\\toprule\n');
fprintf('Temperature $T$ [K]        & %.2f \\\\\n', T);
fprintf('Pressure $P$ [bar]         & %.2f \\\\\n', P);
fprintf('Residence time $\\\\tau$ [s] & %.3f \\\\\n', tau);
fprintf('\\midrule\n');
fprintf('Binding energy $E_b$ [eV]  & %.3f \\\\\n', Eb);
fprintf('Barrier shift $\\\\Delta E_a$ [eV] & %.3f \\\\\n', dEa);
fprintf('\\midrule\n');
fprintf('Yield $Y$                  & %.4f \\\\\n', Y);
fprintf('\\bottomrule\n');
fprintf('\\end{tabular}\n');
fprintf('\\caption{Optimal operating point}\n');
fprintf('\\label{tab:optimum}\n');
fprintf('\\end{table}\n');
fprintf('\n');
end

function tex = write_opt_solution_table(filename, best_proc, best_cat, best_y, varargin)
p = inputParser;
p.addParameter('Caption','Optimal operating point');
p.addParameter('Label','tab:optimum');
p.addParameter('Digits',struct('T',2,'P',2,'tau',3,'Eb',3,'dEa',3,'Y',4));
p.addParameter('ScaleY','unit'); 
p.parse(varargin{:});
opt = p.Results;

T   = best_proc(1);
P   = best_proc(2);
tau = best_proc(3);
Eb  = best_cat(1);
dEa = best_cat(2);
Y   = best_y;

if strcmpi(opt.ScaleY,'percent')
    Ynum  = 100*Y;
    Yunit = '\%';
    Yfmt  = sprintf('%%.%df', opt.Digits.Y);
else
    Ynum  = Y;
    Yunit = '';
    Yfmt  = sprintf('%%.%df', opt.Digits.Y);
end

fmtT   = sprintf('%%.%df', opt.Digits.T);
fmtP   = sprintf('%%.%df', opt.Digits.P);
fmttau = sprintf('%%.%df', opt.Digits.tau);
fmtEb  = sprintf('%%.%df', opt.Digits.Eb);
fmtdEa = sprintf('%%.%df', opt.Digits.dEa);

lines = {
'\begin{table}[t]'
'\centering'
'\begin{tabular}{l r}'
'\toprule'
sprintf('Temperature $T$ [K]        & %s \\\\', sprintf(fmtT,   T))
sprintf('Pressure $P$ [bar]         & %s \\\\', sprintf(fmtP,   P))
sprintf('Residence time $\\tau$ [s] & %s \\\\', sprintf(fmttau, tau))
'\midrule'
sprintf('Binding energy $E_b$ [eV]  & %s \\\\', sprintf(fmtEb,  Eb))
sprintf('Barrier shift $\\Delta E_a$ [eV] & %s \\\\', sprintf(fmtdEa, dEa))
'\midrule'
sprintf('Yield $Y$                  & %s%s \\\\', sprintf(Yfmt,  Ynum), Yunit)
'\bottomrule'
'\end{tabular}'
sprintf('\\caption{%s}', opt.Caption)
sprintf('\\label{%s}',   opt.Label)
'\end{table}'
};

tex = strjoin(lines, newline);

if ~isempty(filename)
    fid = fopen(filename,'w');
    assert(fid>0, 'Cannot open %s for writing.', filename);
    fwrite(fid, tex);
    fclose(fid);
    fprintf('LaTeX table written to %s\n', filename);
end
end
