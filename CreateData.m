% =========================================================================
% SYNTHETIC DATA GENERATOR FOR 2D GRAVITY FORWARD MODELING
% 
% Description: 
% This script generates synthetic 2D gravity data using the Talwani method.
% It creates the forward model, adds Gaussian noise (5%), and saves the 
% observed data to a text file for inversion purposes.
%
% Usage:
% Select the model by changing the 'Modelname' variable ('1' or '2').
% The output will be saved as 'dObs Model-X.txt' in the current directory.
% =========================================================================

clear; close all; clc;

%% === Constants & Grid Setup ===
ncorn = 4;
z0 = 0;              % Topography

dx = 100; dz = dx;   % Box length in meters
m = dx / 2;
nx = 80; nz = 20;    % Number of boxes
M = nx * nz;

% Location of measurement
x = (1:nx) * dx - m;
N = numel(x);

%% === Model Generation ===
Modelname = '2';     % Change to '1' or '2' as per model requirement

rho = zeros(nz, nx);

if strcmp(Modelname, '1')
    start_row = round(nz * 4 / 15);       
    end_row = round(nz * 7 / 15);         
    width = round(nx * 7 / 60);           
    start_col = round((nx - width) / 2);
    rho(start_row:end_row, start_col:start_col + width - 1) = 1;
    
elseif strcmp(Modelname, '2')
    start_row = round(nz * 4 / 15);       
    end_row = round(nz * 7 / 15);         
    start_col = round(nx * 16 / 60);      
    width = round(nx * 7 / 60);          
    rho(start_row:end_row, start_col:start_col + width - 1) = 1;
    
    % === Parameter Dyke ===
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

%% === Generate Grid & Preallocate ===
V = reshape(rho', [], 1);

% Generate grid using meshgrid
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

%% === Kernel Matrix & Forward Calculation ===
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

% Forward Calculation
dObs = A * V;

%% === Add Noise & Save Data ===
% Calculate the standard deviation of noise as 5% of the maximum amplitude of the data.
std_noise = 0.05 * max(abs(dObs)); 
noise = std_noise * randn(size(dObs)); 
dObs_noise = dObs + noise;

% Save Data to TXT
filename = sprintf('dObs Model-%s.txt', Modelname);
data_to_save = [x(:), dObs_noise(:)];
fileID = fopen(filename, 'w');
fprintf(fileID, '%f %f\n', data_to_save');
fclose(fileID);
fprintf('Data was successfully saved to %s\n', filename);

%% === Plotting ===
figure;
set(gcf, 'Position', [150, 100, 800, 500]);

% 1. Plot observed data
subplot(2,1,1);
plot(x/1000, dObs_noise, 'o', 'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k'); hold on;
plot(x/1000, dObs, '-r', 'LineWidth', 1.5); % Garis data tanpa noise untuk referensi
xlabel('Distance (km)', 'FontWeight', 'bold'); 
ylabel('\Delta g (mGal)', 'FontWeight', 'bold');
xlim([min(x)-m max(x)+m]/1000);
legend('Noisy Data (5%)', 'True Forward Data', 'Location', 'best', 'box', 'off');
set(gca, 'LineWidth', 1.5, 'Box', 'on');

% 2. Plot true model
subplot(2,1,2);
hold on;
for i = 1:nz
    for j = 1:nx
        patch([xm(i,j) xm(i,j+1) xm(i+1,j+1) xm(i+1,j)] / 1000, ...
              [zm(i,j) zm(i,j+1) zm(i+1,j+1) zm(i+1,j)] / 1000, ...
              rho(i,j), 'EdgeColor', '#808080');
    end
end
clim([0 1]);
xlabel('Distance (km)', 'FontWeight', 'bold');
ylabel('Depth (km)', 'FontWeight', 'bold'); 
set(gca, 'YDir', 'reverse', 'LineWidth', 1.5);
xlim([min(x) max(x)] / 1000); axis tight;

try
    colormap(gca, viridis);
catch
    colormap(gca, parula);
end

h = colorbar('Position', [0.93 0.11 0.015 0.35], 'Orientation', 'vertical');
set(get(h, 'label'), 'string', 'Density Contrast (g/cc)', 'FontSize', 10);

%% ========================================================================
%  FUNCTIONS SECTION
%  ========================================================================

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