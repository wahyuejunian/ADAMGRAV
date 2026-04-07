% =========================================================================
% 2D GRAVITY FORWARD MODELING AND INVERSION USING ADAM OPTIMIZER
% 
% Description: 
% This script performs 2D gravity forward modeling using the Talwani method 
% and inversion using the Adam optimization algorithm. 
%
% Usage: 
% Ensure 'dObs Model-1.txt' or 'dObs Model-2.txt' is in the same directory.
% Select the model by changing the 'Modelname' variable.
% =========================================================================

clear; close all; clc;

% Tambahkan folder Colormap menggunakan relative path (sesuaikan dengan struktur repo)
% addpath('./Colormap'); 

%% === Constants & Grid Setup ===
ncorn = 4;
z0 = 0;              % Topography

dx = 100; dz = dx;   % Box length in meters
m = dx / 2;
nx = 80; nz = 20;    % Number of boxes
M = nx * nz;

%% === Model Selection ===
Modelname = '1';     % Change to '2' for Model-2

if strcmp(Modelname, '1')
    data = load('dObs Model-1.txt');
    rho = zeros(nz, nx);
    
    start_row = round(nz * 4 / 15);       % Baris awal proporsional
    end_row = round(nz * 7 / 15);         % Baris akhir proporsional
    width = round(nx * 7 / 60);           % Lebar anomali proporsional
    start_col = round((nx - width) / 2);
    
    rho(start_row:end_row, start_col:start_col + width - 1) = 1;

elseif strcmp(Modelname, '2')
    data = load('dObs Model-2.txt');
    rho = zeros(nz, nx);
    
    start_row = round(nz * 4 / 15);       
    end_row = round(nz * 7 / 15);         
    start_col = round(nx * 16 / 60);      
    width = round(nx * 7 / 60);           
    rho(start_row:end_row, start_col:start_col + width - 1) = 1;

    % Parameter Dyke
    start_row_dyke = round(nz * 4 / 15);  
    end_row_dyke = round(nz * 7 / 14);    
    start_col_dyke = round(nx * 40 / 60); 
    width_dyke = round(nx * 4 / 60);      
    rows = start_row_dyke:end_row_dyke;   
    cols = (start_col_dyke:1:start_col_dyke + 1 * (end_row_dyke - start_row_dyke));
    
    for i = 1:length(rows)
        rho(rows(i), cols(i):cols(i) + width_dyke - 1) = 0.7;
    end
else
    error('Modelname must be 1 or 2');
end

x = data(:, 1);
dObs_noise = data(:, 2);
N = length(dObs_noise);

%% === Generate Grid & Kernel Matrix ===
[xm, zm] = meshgrid(0:dx:(nx*dx), 0:dz:(nz*dz));

% Preallocate gz
gz = zeros(N, nz, nx);
for n = 1:N
    for i = 1:nz
        for j = 1:nx
            xmm = [xm(i,j); xm(i,j+1); xm(i+1,j+1); xm(i+1,j)];
            zmm = [zm(i,j); zm(i,j+1); zm(i+1,j+1); zm(i+1,j)];
            gz(n,i,j) = Talwani(x(n), z0, xmm, zmm, ncorn);
        end
    end
end

% Kernel matrix A
A = zeros(N, M);
for n = 1:N
    k = 1;
    for i = 1:nz
        for j = 1:nx
            A(n, k) = gz(n, i, j);
            k = k + 1;
        end
    end
end

%% === Inversion Setup ===
V_inv = zeros(size(reshape(rho', [], 1)));

% Adam Optimizer settings
state = struct;
initial_learning_rate = 1;
learning_rate = initial_learning_rate;
num_iterations = 1000;
trajectory = zeros(length(V_inv), num_iterations);
err = zeros(1, num_iterations);

% Weighting Data and Depth
z = (dz / 2):dz:(nz * dz - dz / 2);
beta = 6;
P = kron(diag(z.^beta), eye(nx));
Wd = diag(1 ./ (1 + abs(dObs_noise)));  % Data weighting

lambda = 1e-5;
scaling_factor_0 = norm(P * A' * Wd * A, 'fro');

rmax = 1; rmin = 0;
tol = 0.04;
decay_factor = 0.01;

%% === Inversion Loop ===
for iter = 1:num_iterations
    residual = dObs_noise - A * V_inv;
    misfit = norm(residual) / sqrt(N);

    penalty = lambda * norm(P * A' * Wd * residual);
    gradient = P * A' * Wd * residual - penalty * V_inv;

    state.alpha = learning_rate;
    [update, state] = Adam(gradient, state);

    V_inv = min(max(V_inv + update, rmin), rmax);  % Constraint clamping
    trajectory(:, iter) = V_inv;
    err(iter) = misfit;

    if misfit < tol
        err(iter+1:end) = []; % Trim unused preallocated array
        break;
    end

    learning_rate = initial_learning_rate / (1 + decay_factor * iter);
end

%% === Results & Evaluation ===
dCal = A * V_inv;
rho_inv = reshape(V_inv, nx, nz)';

rho_true_vector = reshape(rho', [], 1);
rho_inv_vector = reshape(rho_inv', [], 1);
correlation_matrix = corrcoef(rho_true_vector, rho_inv_vector);
correlation_factor = correlation_matrix(1, 2);

err_disp = norm(dObs_noise - dCal) / sqrt(N);

fprintf('Faktor Korelasi (Sintetik vs Inversi): %.4f\n', correlation_factor);
fprintf('Eror RMS: %.4f\n', err_disp);

%% === Plotting ===
figure(1);
set(gcf, 'Position', [100, 20, 800, 600]); 

% 1. Plot observed vs calculated data
subplot(3,1,1);
a1 = plot(x/1000, dObs_noise, 'ob', 'MarkerFaceColor', 'b', 'DisplayName', 'Synthetic data'); hold on;
a2 = plot(x/1000, dCal, '-r', 'LineWidth', 2, 'DisplayName', 'Calculated data');
ylabel('\Delta g (mGal)', 'FontWeight', 'bold');
box off;
xlim([min(x)-m max(x)+m]/1000);
legend([a1, a2]);
set(gca, 'LineWidth', 1.5);

% 2. Plot true model
subplot(3,1,2);
hold on;
for i = 1:nz
    for j = 1:nx
        patch([xm(i,j) xm(i,j+1) xm(i+1,j+1) xm(i+1,j)] / 1000, ...
              [zm(i,j) zm(i,j+1) zm(i+1,j+1) zm(i+1,j)] / 1000, ...
              rho(i,j), 'EdgeColor', '#808080');
    end
end
clim([0 1.5]);
ylabel('Depth (km)', 'FontWeight', 'bold'); 
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5);
xlim([min(x) max(x)]/1000); axis tight;
colormap(gca, jet);

if strcmp(Modelname, '2')
    plot_polygon(dx, dz, start_row, end_row, start_col, width, cols, rows, width_dyke);
elseif strcmp(Modelname, '1')
    Model1(dx, dz, start_row, end_row, start_col, width);
end

% 3. Plot inverted model
subplot(3,1,3);
hold on;
for i = 1:nz
    for j = 1:nx
        patch([xm(i,j) xm(i,j+1) xm(i+1,j+1) xm(i+1,j)] / 1000, ...
              [zm(i,j) zm(i,j+1) zm(i+1,j+1) zm(i+1,j)] / 1000, ...
              rho_inv(i,j), 'EdgeColor', '#808080');
    end
end
clim([0 1.5]);
xlabel('Distance (km)', 'FontWeight', 'bold');
ylabel('Depth (km)', 'FontWeight', 'bold'); 
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5);
xlim([min(x) max(x)]/1000); axis tight;
colormap(gca, jet);

h = colorbar('Position', [0.93 0.108 0.01 0.52], 'Orientation', 'vertical');
set(get(h, 'label'), 'string', 'Density Contrast (g/cc)', 'FontSize', 10);

if strcmp(Modelname, '2')
    plot_polygon(dx, dz, start_row, end_row, start_col, width, cols, rows, width_dyke);
elseif strcmp(Modelname, '1')
    Model1(dx, dz, start_row, end_row, start_col, width);
end

% Save figure
print(gcf, sprintf('Model%s_Result.png', Modelname), '-dpng', '-r400');

% Plot convergence curve
figure(2);
plot(err, 'LineWidth', 1.5);
xlabel('Iteration');
ylabel('Misfit Error');
title('Convergence Curve');
grid on;


%% ========================================================================
%  FUNCTIONS SECTION
%  ========================================================================

function [updates, state] = Adam(gradients, state)
    % ADAM Optimizer
    if nargin == 1
        state = struct;
    end

    if ~isfield(state, 'beta1'), state.beta1 = 0.9; end
    if ~isfield(state, 'beta2'), state.beta2 = 0.999; end
    if ~isfield(state, 'epsilon'), state.epsilon = 1e-8; end
    if ~isfield(state, 'iteration'), state.iteration = 1; end
    if ~isfield(state, 'm'), state.m = zeros(size(gradients)); end
    if ~isfield(state, 'v'), state.v = zeros(size(gradients)); end
    if ~isfield(state, 'alpha'), state.alpha = 1e-2; end

    state.m = state.beta1 * state.m + (1 - state.beta1) * gradients;
    state.v = state.beta2 * state.v + (1 - state.beta2) * gradients.^2;

    mhat = state.m / (1 - state.beta1^state.iteration);
    vhat = state.v / (1 - state.beta2^state.iteration);

    updates = state.alpha * mhat ./ (sqrt(vhat) + state.epsilon);
    state.iteration = state.iteration + 1;
end

function g = Talwani(x0, z0, xcorn, zcorn, ncorn)
    % TALWANI calculates the vertical gravitational attraction of a 2D polygon
    si2mg = 1e5;        % SI to mGal
    km2m = 1e3;         % km to m
    G = 6.673e-11;      % Gravity const, N.m^2/kg^2

    sumG = 0;
    for n = 1:ncorn     
        if n == ncorn
            n2 = 1;
        else
            n2 = n + 1;
        end

        x1 = xcorn(n) - x0;
        z1 = zcorn(n) - z0;
        x2 = xcorn(n2) - x0;
        z2 = zcorn(n2) - z0;

        r1sq = x1^2 + z1^2;
        r2sq = x2^2 + z2^2;

        if r1sq == 0 || r2sq == 0
            error('GPOLY: Field point on corner');
        end

        denom = z2 - z1;
        if denom == 0
            denom = 1e-6;
        end

        alpha = (x2 - x1) / denom;  
        beta = (x1 * z2 - x2 * z1) / denom;

        factor = beta / (1 + alpha^2);
        term1 = 0.5 * (log(r2sq) - log(r1sq));
        term2 = atan2(z2, x2) - atan2(z1, x1);

        sumG = sumG + factor * (term1 - alpha * term2);
    end

    g = 2 * G * sumG * si2mg * km2m;
end

function Model1(dx, dz, start_row, end_row, start_col, width)
    % Draw boundary for Model 1
    lw = 2.5;
    cl = 'r';
    plot([start_col-1 start_col + width-1] * dx / 1000, [start_row-1 start_row-1] * dz / 1000, '-', 'Color', cl , 'LineWidth', lw);
    plot([start_col-1 start_col + width-1] * dx / 1000, [end_row end_row] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw);
    plot([start_col-1 start_col-1] * dx / 1000, [start_row-1 end_row] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw);
    plot([start_col + width-1 start_col + width-1] * dx / 1000, [start_row-1 end_row] * dz / 1000, 'r-', 'LineWidth', lw);
end

function plot_polygon(dx, dz, start_row, end_row, start_col, width, cols, rows, width_dyke)
    % Draw boundaries for Model 2
    lw = 2.5;
    cl = 'r';
    plot([start_col-1 start_col + width-1] * dx / 1000, [start_row-1 start_row-1] * dz / 1000, '-', 'Color', cl , 'LineWidth', lw); 
    plot([start_col-1 start_col + width-1] * dx / 1000, [end_row end_row] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw); 
    plot([start_col-1 start_col-1] * dx / 1000, [start_row-1 end_row] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw); 
    plot([start_col + width-1 start_col + width-1] * dx / 1000, [start_row-1 end_row] * dz / 1000, 'r-', 'LineWidth', lw); 

    for i = 1:length(rows)
        plot([52 57] * dx / 1000, [rows(1)-1 rows(1)-1] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw); 
        plot([57 62] * dx / 1000, [rows(6) rows(6)] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw); 
        plot([cols(i)-1 cols(i)-1] * dx / 1000, [rows(i)-1 rows(i)] * dz / 1000, '-', 'Color', cl, 'LineWidth', 1.5); 
        plot([cols(i) + width_dyke-1 cols(i) + width_dyke-1] * dx / 1000, [rows(i)-1 rows(i)] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw); 
        plot([cols(i)+ width_dyke-2 cols(i) + width_dyke-1] * dx / 1000, [rows(i)-1 rows(i)-1] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw);
        plot([cols(i)-1 cols(i)] * dx / 1000, [rows(i) rows(i)] * dz / 1000, '-', 'Color', cl, 'LineWidth', lw);
    end
end