function [hist_best, iter_y, n_iter, best_proc, best_cat, best_y] = ...
          bo_outer_catalyst__fmincon_inner_process_pureEI(doPlot)
%% Handle optional input
if nargin < 1 || isempty(doPlot)
    doPlot = true;   % default behavior
end

% Hybrid optimization:
%   Outer: Bayesian Optimization (MATLAB fitrgp + pure EI) over catalyst [Eb, dEa]
%   Inner: fmincon over process [T, P, tau]
% Consecutive reactions A -> B -> C in a CSTR (feed = A only).
% Objective: MAXIMIZE Yield_B 

 rng(26362063)

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');

% -------------------- Bounds --------------------
% Process:   T[K],  P[bar], tau[s]
lb_proc = [450,  1, 0.1];
ub_proc = [700, 20, 5.0];

% Catalyst:  Eb[eV], dEa[eV]
lb_cat  = [-2.0, -0.2];
ub_cat  = [ 0.0,  0.2];
d_cat   = 2;

% Normalization helpers for catalyst
normC  = @(Xc) (Xc - lb_cat) ./ (ub_cat - lb_cat + eps);
denorm = @(Zn) bsxfun(@plus, lb_cat, bsxfun(@times, Zn, (ub_cat - lb_cat)));

% -------------------- BO settings --------------------
n_init = 28;      % number of initial catalyst points
n_iter = 100;     % BO iterations
acq_M  = 2000;    % candidate bank size (before edges/jitter)

% -------------------- Initial design: pure LHS --------------------
U0 = lhs_unit(n_init, d_cat);   % U0 in [0,1]^2 (Eb, dEa)
Xc = denorm(U0);                % map to real catalyst units

yc        = zeros(n_init,1);
Xproc_opt = zeros(n_init,3);

Ntot        = n_init + n_iter;
log_cat     = zeros(Ntot, 2);
log_proc    = zeros(Ntot, 3);
log_y       = zeros(Ntot, 1);
log_is_init = false(Ntot,1);
log_mu      = nan(Ntot, 1);
log_sigma   = nan(Ntot, 1);
log_ei      = nan(Ntot, 1);

cursor = 0;
for i = 1:n_init
    [yc(i), Xproc_opt(i,:)] = inner_process_opt_consecutive(Xc(i,:), lb_proc, ub_proc);
    cursor = cursor + 1;
    log_cat(cursor,:)   = Xc(i,:);
    log_proc(cursor,:)  = Xproc_opt(i,:);
    log_y(cursor)       = yc(i);
    log_is_init(cursor) = true;
end

[best_y, ib] = max(yc);
best_cat     = Xc(ib,:);
best_proc    = Xproc_opt(ib,:);
hist_best    = zeros(n_iter,1);
iter_y       = zeros(n_iter,1);

fprintf('---- Initial evaluations (n=%d) ----\n', n_init);
print_log_header();
for k = 1:cursor
    print_log_row(k, log_is_init(k), log_cat(k,:), log_proc(k,:), log_y(k), NaN, NaN, NaN);
end

% -------------------- BO loop (over catalyst) --------------------
tstart = tic;
for t = 1:n_iter
    % --- Train GP surrogate in catalyst space using MATLAB fitrgp ---
    Xtrain_n = normC(Xc);
    gprMdl = fitrgp(Xtrain_n, yc, ...
        'KernelFunction','ardsquaredexponential', ...
        'Standardize',true);

    params = gprMdl.KernelInformation.KernelParameters;
    ell    = params(1:d_cat);
    sf     = params(end);
    sn_eff = gprMdl.Sigma;

    fprintf('Iter %2d | ell=[%.3g %.3g], sf=%.3g, sn=%.3g | std(y)=%.3g\n', ...
        t, ell(1), ell(2), sf, sn_eff, std(yc));

    % ----- Candidate bank (normalized) -----
    Nc      = acq_M;
    U_lhs   = lhs_unit(Nc, 2);
    U_rnd   = rand(Nc, 2);
    best_n  = normC(best_cat);
    U_jit   = min(max(best_n + 0.10*randn(round(0.75*Nc),2), 0), 1);
    corners = [0 0; 0 1; 1 0; 1 1];
    edges   = [linspace(0,1,50)' zeros(50,1); linspace(0,1,50)' ones(50,1); ...
               zeros(50,1) linspace(0,1,50)'; ones(50,1) linspace(0,1,50)'];

    Xcand_n = [U_lhs; U_rnd; U_jit; corners; edges];

    % ----- Predict on candidates -----
    [mu_c, s2_c] = predict(gprMdl, Xcand_n);
    s_c = sqrt(max(s2_c,0));

    % ===== Pure EI acquisition (no fallbacks) =====
    f_best = max(yc);   % current best observed yield
    xi     = 0.0;       % no extra exploration offset
    EI     = expected_improvement_with_xi(mu_c, s_c, f_best, xi);

    [EImax, idx] = max(EI);
    xcat_new_n   = Xcand_n(idx,:);
    xcat_new     = denorm(xcat_new_n);

    mu_at_x = mu_c(idx);
    s_at_x  = s_c(idx);
    ei_val  = EI(idx);

    fprintf('  Select EI maximizer | EImax=%.3g (f_best=%.3g)\n', EImax, f_best);

    % INNER: fmincon over process
    [y_new, xproc_new] = inner_process_opt_consecutive(xcat_new, lb_proc, ub_proc);

    % Update data
    Xc         = [Xc; xcat_new];
    yc         = [yc; y_new];
    Xproc_opt  = [Xproc_opt; xproc_new];

    if y_new > best_y
        best_y   = y_new;
        best_cat = xcat_new;
        best_proc= xproc_new;
    end
    hist_best(t) = best_y;
    iter_y(t)    = y_new;

    % Log row
    cursor = cursor + 1;
    log_cat(cursor,:)  = xcat_new;
    log_proc(cursor,:) = xproc_new;
    log_y(cursor)      = y_new;
    log_mu(cursor)     = mu_at_x;
    log_sigma(cursor)  = s_at_x;
    log_ei(cursor)     = ei_val;

    print_log_row(cursor, false, xcat_new, xproc_new, y_new, mu_at_x, s_at_x, ei_val);
end
telapsed = toc(tstart);

% -------------------- Results summary --------------------
fprintf('\n=== Hybrid BO(fmincon) with pure EI complete ===\n');
fprintf('Time elapsed = %.6f s \n', telapsed);
fprintf('Best Yield f = %.6f\n', best_y);
fprintf('  Catalyst*   : E_bind = %.4f eV,  dEa = %.4f eV\n', best_cat(1), best_cat(2));
fprintf('  Process*    : T = %.3f K,  P = %.3f bar,  tau = %.3f s\n', best_proc(1), best_proc(2), best_proc(3));

% Final GP for all the plotting (fitrgp again with full data)
Xtrain_n = normC(Xc);
gprMdl   = fitrgp(Xtrain_n, yc, ...
    'KernelFunction','ardsquaredexponential', ...
    'Standardize',true);

params = gprMdl.KernelInformation.KernelParameters;
ell    = params(1:d_cat);
sf     = params(end);
sn_eff = gprMdl.Sigma;
fprintf('\nFinal GP hyperparameters: ell=[%.4f %.4f], sf=%.4f, sn=%.4f\n', ell(1), ell(2), sf, sn_eff);

% -------------------- Top-K table --------------------
K = min(10, size(Xc,1));
[sorted_y, sIdx] = sort(yc, 'descend');
fprintf('\nTop %d designs (by yield):\n', K);
fprintf('Rank |   f(Y)   | Eb (eV)   dEa (eV) |   T (K)     P (bar)   tau (s)\n');
fprintf('--------------------------------------------------------------------\n');
for k = 1:K
    i = sIdx(k);
    fprintf('%4d | %8.4f | %8.3f  %8.3f | %8.2f  %8.2f  %8.2f\n', ...
        k, sorted_y(k), Xc(i,1), Xc(i,2), Xproc_opt(i,1), Xproc_opt(i,2), Xproc_opt(i,3));
end

% -------------------- Visualizations --------------------
tt = 1:n_iter;

if doPlot
    % 1) Progress
    figure('Name','BO Progress and per-iteration objective (pure EI)','Color','w');
    set(groot, 'DefaultAxesFontName', 'Helvetica');
    set(groot, 'DefaultAxesFontSize', 12);
    set(groot, 'DefaultTextFontName', 'Helvetica');
    set(groot, 'DefaultTextFontSize', 12);

    subplot(2,1,1);
    plot(tt, hist_best*100, 'k-', 'LineWidth',1.5); grid on;
    xlabel('BO iteration');
    ylabel('Best $Y_B$ (\%)','Interpreter','latex');
    title('Best-so-far');

    subplot(2,1,2);
    stem(tt, iter_y*100, 'filled'); grid on;
    xlabel('BO iteration');
    ylabel('$Y_B$ at iteration (\%)','Interpreter','latex');
    title('Per-iteration objective');
end

% ---------------- CSV Export ----------------
exportData = table( ...
    tt(:), ...
    hist_best(:)*100, ...
    iter_y(:)*100, ...
    'VariableNames', {'Iteration','BestSoFar_pct','PerIter_pct'} );

outDir = 'exports';
if ~exist(outDir,'dir'); mkdir(outDir); end

outFile = fullfile(outDir, sprintf('Hybrid_BO_progress_pureEI_%s.csv', ...
                                   datestr(now,'yyyymmdd_HHMMSS')));
writetable(exportData, outFile);
fprintf('CSV exported: %s\n', outFile);

if doPlot
    % 2) Catalyst space scatter
    figure('Name','Catalyst evaluations colored by yield (pure EI)','Color','w');
    scatter3(Xc(:,1), Xc(:,2), yc, 36, yc, 'filled'); grid on; colorbar;
    xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); zlabel('$Y_B$');
    title('Observed yield across catalyst space');
    hold on; 
    plot3(best_cat(1), best_cat(2), best_y, 'rp', 'MarkerSize',12, 'MarkerFaceColor','r'); 
    hold off;

    % 3) 2D μ, σ, EI maps
    figure('Name','Catalyst space: \mu, \sigma, EI maps (pure EI)','Color','w');
    visualize_mu_sigma_ei_maps_muBands_norm(gprMdl, lb_cat, ub_cat, Xc, yc, best_cat, normC);

    % 4) 1D profiles in catalyst space (normalized)
    figure('Name','1D profiles in catalyst space (normalized, pure EI)','Color','w');
    visualize_catalyst_1D_norm(gprMdl, lb_cat, ub_cat, yc, ...
        {'$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'}, normC);

    % 5) 1D profiles in catalyst space (real axis + CSV export)
    figure('Name','1D profiles in catalyst space (real units, pure EI)','Color','w');
    visualize_catalyst_1D_real(gprMdl, lb_cat, ub_cat, yc, ...
        {'$E_{\mathrm{bind}}$ (eV)','$\Delta E_a$ (eV)'}, normC);

    print_opt_solution_table_columns(best_proc, best_cat, best_y)
end

end % ===== main =====

%% --------- LaTeX table  -----------------------------------------
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

%% ========================= INNER: fmincon over process ===================
function [best_f, best_xproc] = inner_process_opt_consecutive(xcat, lb_proc, ub_proc)
% Maximize Yield_B(T,P,tau; Eb,dEa) for fixed catalyst xcat = [Eb dEa]
Eb  = xcat(1);  dEa = xcat(2);

obj_proc = @(xp) obj_and_grad_process_consecutive(xp, Eb, dEa);

opts = optimoptions('fmincon', ...
    'Algorithm','sqp', 'Display','off', ...
    'MaxFunctionEvaluations', 3e3, ...
    'StepTolerance', 1e-9, 'OptimalityTolerance', 1e-9, ...
    'SpecifyObjectiveGradient', true);

% Multistart
n_starts = 6;
X0 = zeros(n_starts,3);
X0(1,:) = 0.5*(lb_proc + ub_proc);
for i=2:n_starts
    X0(i,:) = lb_proc + rand(1,3).*(ub_proc - lb_proc);
end

best_f    = -Inf;
best_xproc= X0(1,:);
for i=1:n_starts
    x0 = X0(i,:);
    [x_opt, fneg] = fmincon(obj_proc, x0, [],[],[],[], lb_proc, ub_proc, [], opts);
    f_val = -fneg;  % back to maximize
    if f_val > best_f
        best_f    = f_val;
        best_xproc= x_opt;
    end
end
end

function [fneg, gradneg] = obj_and_grad_process_consecutive(xp, Eb, dEa)
% Returns negative objective and its gradient (for fmincon)
T = xp(1); P = xp(2); tau = xp(3);

R = 8.314;   A1=1e7;   A2=3e6;   Pref=8.0;
beta1=1.0;   beta2=0.35;
Eb0=-1.0;    sigmaE=0.40;   alpha=0.6;
K0=0.06;     kappa=3.0;

volc  = exp(-0.5*((Eb - Eb0)/sigmaE).^2);
dvolc_dEb = -(Eb - Eb0)/(sigmaE^2) * volc;

Kads  = K0 * exp(kappa*(Eb - Eb0));
dKads_dEb = kappa * Kads;

denL  = (1 + Kads*P)^2;
ddenL_dP  = 2*(1 + Kads*P)*Kads;
ddenL_dEb = 2*(1 + Kads*P)*P*dKads_dEb;

Ea1 = 80000 - 20000*dEa;   dEa1_ddEa = -20000;
Ea2 = 95000 +  5000*dEa;   dEa2_ddEa =  5000;

pf1 = exp(beta1*log(P/Pref));
pf2 = exp(beta2*log(P/Pref));
dpf1_dP = pf1 * (beta1 / P);
dpf2_dP = pf2 * (beta2 / P);

phi1 = exp(-Ea1/(R*T));
phi2 = exp(-Ea2/(R*T));
dphi1_dT   = phi1 * (Ea1/(R*T^2));
dphi2_dT   = phi2 * (Ea2/(R*T^2));
dphi1_ddEa = phi1 * (-1/(R*T)) * dEa1_ddEa;
dphi2_ddEa = phi2 * (-1/(R*T)) * dEa2_ddEa;

k1 = A1 * volc            * phi1 * pf1 / denL;
k2 = A2 * (1 - alpha*volc)* phi2 * pf2 / denL;

dk1_dT   = k1 * (dphi1_dT/phi1);
dk1_dP   = k1 * (dpf1_dP/pf1) - A1*volc*phi1*pf1 * (ddenL_dP)/(denL^2);
dk1_dEb  = k1 * (dvolc_dEb/volc) - A1*volc*phi1*pf1 * (ddenL_dEb)/(denL^2);
dk1_ddEa = k1 * (dphi1_ddEa/phi1);

invVolc = (1 - alpha*volc);
dinvVolc_dEb = -alpha * dvolc_dEb;

dk2_dT   = k2 * (dphi2_dT/phi2);
dk2_dP   = k2 * (dpf2_dP/pf2) - A2*invVolc*phi2*pf2 * (ddenL_dP)/(denL^2);
dk2_dEb  = A2*dinvVolc_dEb*phi2*pf2/denL - A2*invVolc*phi2*pf2*(ddenL_dEb)/(denL^2);
dk2_ddEa = k2 * (dphi2_ddEa/phi2);

a = k1*tau;
b = (1 + k1*tau);
c = (1 + k2*tau);
Y = a/(b*c);

da_dT   = dk1_dT * tau;   da_dP   = dk1_dP * tau;   da_dtau = k1;
db_dT   = dk1_dT * tau;   db_dP   = dk1_dP * tau;   db_dtau = k1;
dc_dT   = dk2_dT * tau;   dc_dP   = dk2_dP * tau;   dc_dtau = k2;

den = (b^2)*(c^2);
coeff = 1/den;

dY_dT   = coeff*( da_dT*b*c - a*( db_dT*c + b*dc_dT ) );
dY_dP   = coeff*( da_dP*b*c - a*( db_dP*c + b*dc_dP ) );
dY_dtau = coeff*( da_dtau*b*c - a*( db_dtau*c + b*dc_dtau ) );

fneg    = -Y;
gradneg = -[dY_dT; dY_dP; dY_dtau];
end

%% ========================= Acquisition (EI with xi) =====================
function EI = expected_improvement_with_xi(mu, s, f_ref, xi)
% EI for maximization with reference f_ref and exploration xi, using
% MATLAB's normcdf / normpdf.
s    = max(s, 1e-12);
impr = mu - f_ref - xi;
z    = impr ./ s;

Phi  = normcdf(z);
phi  = normpdf(z);

EI   = impr .* Phi + s .* phi;
EI(s <= 1e-12) = 0;
EI   = max(EI, 0);
end

%% ========================= Plotting helpers =============================
function visualize_mu_sigma_ei_maps_muBands_norm(gprMdl, lb_cat, ub_cat, Xc, yc, best_cat, normC)
N = 160;
Ei = linspace(lb_cat(1), ub_cat(1), N);
Di = linspace(lb_cat(2), ub_cat(2), N);
[EE, DD] = meshgrid(Ei, Di);
Xs_real  = [EE(:), DD(:)];
Xs_norm  = normC(Xs_real);

[mu, s]  = gpr_predict(gprMdl, Xs_norm);
EI       = expected_improvement_with_xi(mu, s, max(yc), 0.05);

MU   = reshape(mu, size(EE));
SIG  = reshape(s,  size(EE));
EI2  = reshape(EI, size(EE));
MUplus  = MU + 2*SIG;
MUminus = MU - 2*SIG;

subplot(2,3,1); safe_fieldplot(EE, DD, MU, 18);
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('$\mu$'); hold on;
scatter(Xc(:,1), Xc(:,2), 16, 'k', 'filled', 'MarkerFaceAlpha',0.5);
plot(best_cat(1), best_cat(2), 'rp','MarkerSize',10,'MarkerFaceColor','r'); hold off;

subplot(2,3,2); safe_fieldplot(EE, DD, MUplus, 18);
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('$\mu + 2\sigma$'); hold on;
scatter(Xc(:,1), Xc(:,2), 16, 'k', 'filled', 'MarkerFaceAlpha',0.5);
plot(best_cat(1), best_cat(2), 'rp','MarkerSize',10,'MarkerFaceColor','r'); hold off;

subplot(2,3,3); safe_fieldplot(EE, DD, MUminus, 18);
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('$\mu - 2\sigma$'); hold on;
scatter(Xc(:,1), Xc(:,2), 16, 'k', 'filled', 'MarkerFaceAlpha',0.5);
plot(best_cat(1), best_cat(2), 'rp','MarkerSize',10,'MarkerFaceColor','r'); hold off;

subplot(2,3,4); safe_fieldplot(EE, DD, SIG, 18);
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('$\sigma$'); hold on;
scatter(Xc(:,1), Xc(:,2), 16, 'k', 'filled', 'MarkerFaceAlpha',0.5);
plot(best_cat(1), best_cat(2), 'rp','MarkerSize',10,'MarkerFaceColor','r'); hold off;

subplot(2,3,5); safe_fieldplot(EE, DD, EI2, 18);
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('EI'); hold on;
scatter(Xc(:,1), Xc(:,2), 16, 'k', 'filled', 'MarkerFaceAlpha',0.5);
plot(best_cat(1), best_cat(2), 'rp','MarkerSize',10,'MarkerFaceColor','r'); hold off;

subplot(2,3,6);
scatter(Xc(:,1), Xc(:,2), 28, yc, 'filled'); colorbar; grid on;
xlabel('$E_{\mathrm{bind}}$ (eV)'); ylabel('$\Delta E_a$ (eV)'); title('Observed $Y_B$');
hold on; plot(best_cat(1), best_cat(2), 'kp', 'MarkerSize',10, 'MarkerFaceColor','w'); hold off;
end

function visualize_catalyst_1D_norm(gprMdl, lb_cat, ub_cat, yc, labels, normC)
N       = 250;
f_best  = max(yc);
mid_real= 0.5*(lb_cat + ub_cat);

for k = 1:2
    gridk = linspace(lb_cat(k), ub_cat(k), N)';
    Xs_real = repmat(mid_real, N, 1); 
    Xs_real(:,k) = gridk;
    Xs = normC(Xs_real);

    [mu, s] = gpr_predict(gprMdl, Xs);
    EI      = expected_improvement_with_xi(mu, s, f_best, 0.05);

    subplot(2,2,k); hold on; grid on;
    fill([gridk; flipud(gridk)], [mu-2*s; flipud(mu+2*s)], [0.85 0.85 0.85], 'EdgeColor','none');
    plot(gridk, mu, 'k-', 'LineWidth',1.25);
    xlabel(labels{k}); ylabel('$GP \mu \pm 2\sigma$');

    subplot(2,2,2+k); plot(gridk, EI, 'k-', 'LineWidth',1.25); grid on;
    xlabel(labels{k}); ylabel('EI'); title(sprintf('EI along %s', labels{k}));
end
end

function visualize_catalyst_1D_real(gprMdl, lb_cat, ub_cat, fvals, labels, normC)
% Plots GP mean ±2σ (left) and EI (right) vs catalyst variables (real axis),
% and exports a CSV similar to your original helper.

tile_w_in = 3; tile_h_in = 2;
nCols = 2; nRows = 1;
fig_w_in = nCols * tile_w_in;
fig_h_in = nRows * tile_h_in;

N        = 300;
f_best   = max(fvals);
mid_real = 0.5*(lb_cat + ub_cat);

figure(gcf); % use current figure
set(gcf, 'Units','inches', 'Position',[1 1 fig_w_in fig_h_in]);
t = tiledlayout(nRows, nCols, 'TileSpacing','compact', 'Padding','compact');

xi_used = 0.10;
series  = cell(2,1);

for k = 1:2
    x_real   = linspace(lb_cat(k), ub_cat(k), N)';  
    Xs_real  = repmat(mid_real, N, 1);
    Xs_real(:,k) = x_real;

    Xs_n = normC(Xs_real);
    [mu_raw, s_raw] = gpr_predict(gprMdl, Xs_n);
    EI_raw = expected_improvement_with_xi(mu_raw, s_raw, f_best, xi_used);

    mu = 100*mu_raw;   s = 100*s_raw;   EI = 100*EI_raw;

    ax = nexttile(t); hold(ax,'on'); grid(ax,'on');

    yyaxis(ax,'left');
    fill(ax, [x_real; flipud(x_real)], [mu - 2*s; flipud(mu + 2*s)], ...
         [0.88 0.88 0.88], 'EdgeColor','none');
    plot(ax, x_real, mu, 'k-', 'LineWidth', 1.25);
    ylabel(ax, 'GP $\mu \pm 2\sigma$ (\%)', 'Interpreter','latex');

    yyaxis(ax,'right');
    plot(ax, x_real, EI, '--', 'LineWidth', 1.25);
    ylabel(ax, 'EI (\%)', 'Interpreter','latex');

    xlabel(ax, labels{k});
    dx = ub_cat(k) - lb_cat(k);
    xlim(ax, [lb_cat(k) - 0.05*dx, ub_cat(k) + 0.05*dx]);

    [~, idx_max] = max(mu);
    x_best_tile = x_real(idx_max);
    xline(ax, x_best_tile, ':','LineWidth',2);

    box(ax,'on'); hold(ax,'off');

    series{k} = table( ...
        repmat(string(labels{k}), N, 1), ...
        repmat(k,                 N, 1), ...
        x_real, ...
        mu_raw, s_raw, EI_raw, ...
        mu, s, EI, ...
        repmat(x_best_tile,        N, 1), ...
        repmat(xi_used,            N, 1), ...
        'VariableNames', {'Variable','Subplot','X_real', ...
                          'Mu_raw','Sigma_raw','EI_raw', ...
                          'Mu_pct','Sigma_pct','EI_pct', ...
                          'X_best_mu','xi'});
end

exportData = vertcat(series{:});
outDir = 'exports';
if ~exist(outDir,'dir'); mkdir(outDir); end
outFile = fullfile(outDir, sprintf('Hybrid_BO_pureEI_%s.csv', ...
                                   datestr(now,'yyyymmdd_HHMMSS'))); %#ok<NASGU>
writetable(exportData, outFile);
fprintf('Hybrid_BO %s\n', outFile);

set(findall(gcf,'-property','FontName'),'FontName','Helvetica');
set(findall(gcf,'-property','FontSize'),'FontSize',12);
end

function safe_fieldplot(X, Y, Z, nlevels)
if std(Z(:)) < 1e-12
    imagesc('XData', X(1,:), 'YData', Y(:,1), 'CData', Z);
    set(gca,'YDir','normal'); axis tight; grid on; colorbar;
else
    contourf(X, Y, Z, nlevels, 'LineStyle','none'); grid on; colorbar;
end
end

%% ========================= utilities =============================
function [mu, s] = gpr_predict(gprMdl, Xn)
% Wrapper to get mean and std from MATLAB's GPR model
[mu, s2] = predict(gprMdl, Xn);
s = sqrt(max(s2, 0));
end

function A = lhs_unit(n, d)
A = zeros(n, d);
for j = 1:d
    p = randperm(n)';                      
    A(:, j) = ((p - 1) + rand(n,1)) / n;   
end
end

function print_log_header()
fprintf('\n%-5s %-6s | %-11s %-11s || %-9s %-9s %-9s || %-9s %-9s %-9s\n', ...
    'Idx','Init?','Eb (eV)','dEa (eV)','T (K)','P (bar)','tau (s)','f','mu','sigma/EI');
fprintf('%s\n', repmat('-',1,94));
end

function print_log_row(idx, is_init, cat, proc, fval, mu_at_x, s_at_x, ei)
tag = tern(is_init,'init','BO  ');
if isnan(mu_at_x) || isnan(s_at_x) || isnan(ei)
    fprintf('%-5d %-6s | %11.3f %11.3f || %9.2f %9.2f %9.2f || %9.4f %9s %9s\n', ...
        idx, tag, cat(1), cat(2), proc(1), proc(2), proc(3), fval, '-', '-');
else
    fprintf('%-5d %-6s | %11.3f %11.3f || %9.2f %9.2f %9.2f || %9.4f %9.4f %9.4f (EI=%8.4f)\n', ...
        idx, tag, cat(1), cat(2), proc(1), proc(2), proc(3), fval, mu_at_x, s_at_x, ei);
end
end

function s = tern(cond, a, b)
if cond, s = a; else, s = b; end
end
