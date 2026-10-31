function out = fene_elongation_fixed_gamma(gammaDot, edotIdx, outDir)
%FENE_ELONGATION_SINGLE  Simulate Rouse–FENE chain(s) under planar extensional flow.
%
% Usage (MATLAB):
%   out = fene_elongation_single(3e-4, 2);                 % save to current dir
%   out = fene_elongation_single(3e-4, 2, '/scratch/...'); % save to outDir
%
% Usage (SLURM batch):
%   matlab -batch "fene_elongation_single(3e-4,'/path')"
%
% Saves:
%   Qsave (N x m x 3 x Nsnap) and tsave (1 x Nsnap) to a MAT-file (v7.3).

if nargin < 2 || isempty(gammaDot)
    error('gammaDot must be provided, e.g. fene_elongation_single(3e-4).');
end
if nargin < 3 || isempty(outDir)
    outDir = pwd;
end

if ~isfolder(outDir)
    mkdir(outDir);
end

% edotIdx saves the second parameter of the elongation rate.  (edotIdx - 4)/2 * 0.1 * gammaDot 

edot = (edotIdx - 2) * 1e-5;


%% ---------------------- Parameters ----------------------
params.N      = 14;           % number of bonds (N); beads = N+1
params.m      = 1600;         % number of independent chains
params.dt     = 1e-2;        % time step
params.Nt     = 1e7;      % total steps
params.Ns     = 10000;        % save every Ns steps

params.gamma  = 170.0;       % friction
params.kBT    = 0.25;        % thermal energy

% FENE parameters
params.H      = 0.1;         % spring constant
params.Q0     = 2.3;         % max extension

% Flow: planar extensional flow (trace zero)
params.kappa = [gammaDot 0 0; 0 -gammaDot+edot 0; 0 0 -edot];


% RNG: default seed + SLURM array task id (if present), so array jobs differ.
taskIdStr = getenv('SLURM_ARRAY_TASK_ID');
taskId = str2double(taskIdStr);
if ~isnan(taskId)
    params.seed = 1 + taskId;
else
    params.seed = 1;
end
rng(params.seed, 'twister');

% Simple deterministic filename based on SLURM array index
arrayIdStr = getenv('SLURM_ARRAY_TASK_ID');
idx = str2double(arrayIdStr);

if ~isnan(idx)
    idxTag = sprintf('%02d', idx);     % "00".."11"
else
    % Local fallback if not running under SLURM array
    idxTag = sprintf('%02d', round(gammaDot / 5e-5));
end

baseName = sprintf('Qsave_gdot_%s.mat', idxTag);
params.saveFile = fullfile(outDir, baseName);


% Optional log
logName = strrep(baseName, '.mat', '.log');
diary(fullfile(outDir, logName));
fprintf('Starting fene_elongation_single at %s\n', datestr(now));
fprintf('gammaDot = %.6e, edDot = %.6e, outDir = %s\n', gammaDot, edot, outDir);
fprintf('saveFile = %s\n', params.saveFile);

%% ---------------------- Precompute A and its factor ----------------------
A = rouse_matrix(params.N);               % sparse N x N
L = chol(A, 'lower');                     % L*L' = A

%% ---------------------- Initialize Q ----------------------
initFile = fullfile(fileparts(mfilename('fullpath')), 'init.mat');
if ~isfile(initFile)
    initFile = '/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_170/init.mat';
end
Qinit = load(initFile);
Q = double(Qinit.Qsave(:,:,:,end));


%% ---------------------- Allocate / prepare saving ----------------------
Nsnap = floor(params.Nt / params.Ns) + 1;
tsave = zeros(1, Nsnap);

Qsave = zeros(params.N, params.m, 3, Nsnap, 'single');


% Save initial snapshot
snapIdx = 1;
tsave(snapIdx) = 0.0;
Qsave(:,:,:,snapIdx) = Q;


%% ---------------------- Time stepping ----------------------
sqrt_dt   = sqrt(params.dt);
noisePref = sqrt(2*params.kBT/params.gamma);

% ---- Relaxation phase (NO flow) ----

tStart = tic;
for n = 1:params.Nt
    t = n * params.dt;

    % ---- Flow term: kappa Q ----
    Qflat  = reshape(Q, [], 3);                 % (3 x (N*m))
    KQflat = Qflat * params.kappa';
    KQ     = reshape(KQflat, params.N, params.m, 3);

    % ---- Spring gradient gradU(Q): (3 x N x m) ----
    gradU  = fene_grad(Q, params.H, params.Q0); %  (N x m x 3)
    AgradU = pagemtimes_sparse(A, gradU);
    dW     = pagemtimes_sparse(L, randn(params.N, params.m,3));
    dQ_stoch = noisePref * sqrt_dt * dW;

    % ---- Euler–Maruyama update ----
    Q = Q + params.dt * (KQ - (1/params.gamma) * AgradU) + dQ_stoch;

    % ---- Save snapshot ----
    if mod(n, params.Ns) == 0
        snapIdx = snapIdx + 1;
        tsave(snapIdx) = t;
        Qsave(:,:,:,snapIdx) = Q;
    end

    % Lightweight progress every ~5% (prints to SLURM output)
    if mod(n, max(1, floor(params.Nt/100))) == 0
        fprintf('Progress: %6.1f%% (step %d/%d), elapsed %.1fs\n', ...
            100*n/params.Nt, n, params.Nt, toc(tStart));
    end
end

%% ---------------------- Outputs ----------------------
out.params = params;
out.A = A; out.L = L;
out.tsave = tsave;

out.saveFile = params.saveFile;
%save(params.saveFile, 'Qsave', 'tsave', 'params', '-v7.3');
save(params.saveFile, 'Qsave', '-v7.3');


fprintf('Simulation complete. Total wall time: %.2f s\n', toc(tStart));
diary off;

end

%% ===== Helper: Rouse matrix (open chain) =====
function A = rouse_matrix(N)
e = ones(N,1);
A = spdiags([-e, 2*e, -e], [-1,0,1], N, N);
end

%% ===== Helper: FENE gradient (vectorized) =====
function gradU = fene_grad(Q, H, Q0)
q = vecnorm(Q,2,3);              % (N x m)
den = max( 1 - (q./Q0).^2 ,  1e-2);
alpha = H ./ den;
gradU = alpha .* Q;                  % implicit expansion (N x m x 3)
end

function C = pagemtimes_sparse(A,B)
C=zeros(size(B));
for i =1:size(B,3)
    C(:,:,i) = A * B(:,:,i);
end 
end