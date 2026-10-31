%% ---------------------- Parameters ----------------------
params.N      = 6;           % number of bonds (N); beads = N+1
params.m      = 1600;         % number of independent chains
params.dt     = 1e-2;        % time step
params.Ns     = 10000;        % save every Ns steps
params.Nr     = 2000000;       % relaxation steps with NO flow (set 0 to disable)

params.gamma  = 170.0;       % friction
params.kBT    = 0.25;        % thermal energy

% FENE parameters
params.H      = 0.1;         % spring constant
params.Q0     = 2.3;         % max extension
params.seed   = 1234;
rng(params.seed, 'twister');

saveName = fullfile(fileparts(mfilename('fullpath')), 'init.mat');

%% ---------------------- Precompute A and its factor ----------------------
A = rouse_matrix(params.N);               % sparse N x N
L = chol(A, 'lower');                     % L*L' = A

%% ---------------------- Initialize Q ----------------------
Q = 0.45 * sqrt(params.kBT/params.H) * randn(params.N, params.m,3);

%% ---------------------- Allocate / prepare saving ----------------------
Nsnap = floor(params.Nr / params.Ns) + 1;
Qsave = zeros(params.N, params.m, 3, Nsnap, 'single');

% Save initial snapshot
snapIdx = 1;
Qsave(:,:,:,snapIdx) = Q;


%% ---------------------- Time stepping ----------------------
sqrt_dt   = sqrt(params.dt);
noisePref = sqrt(2*params.kBT/params.gamma);

% ---- Relaxation phase (NO flow) ----
tStart = tic;
for n = 1:params.Nr
    % Spring gradient
    gradU  = fene_grad(Q, params.H, params.Q0); %  (N x m x 3)
    AgradU = pagemtimes_sparse(A, gradU);
    dW     = pagemtimes_sparse(L, randn(params.N, params.m,3));
    dQ_stoch = noisePref * sqrt_dt * dW;

    % Update WITHOUT flow term
    Q = Q + params.dt * (-(1/params.gamma) * AgradU) + dQ_stoch;

    if mod(n, params.Ns) == 0
        snapIdx = snapIdx + 1;
        Qsave(:,:,:,snapIdx) = Q;
    end

    % Lightweight progress every ~5% (prints to SLURM output)
    if mod(n, max(1, floor(params.Nr/100))) == 0
        yr = squeeze(sum(Q,1))/sqrt(params.N)/sqrt(0.7453349210778061); % [m x 3]
        c0 = mean(yr.*yr,'all');

        fprintf('Progress: %2.1f%% (step %d/%d), elapsed %.1fs, c0 %2.5f\n', ...
            100*n/params.Nr, n, params.Nr, toc(tStart), c0);
    end
end

save(saveName, 'Qsave', '-v7.3');


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