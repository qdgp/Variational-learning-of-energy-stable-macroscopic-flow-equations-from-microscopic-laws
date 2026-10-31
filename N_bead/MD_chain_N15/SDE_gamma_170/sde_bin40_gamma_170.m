clear
Np = 1600;

train_rr = cell(1,1,1,200);
train_ff = cell(1,1,1,200);
flag =1;

Folder = [fullfile(fileparts(mfilename('fullpath')), 'results_v4'), filesep];
if ~isfolder(Folder)
    Folder = '/mnt/gs21/scratch/gepei/DeePN_chain_N15/SDE_gamma_170/results_v4/';
end

parfor flag = 1:200
    i = floor((flag - 1) / 40);
    j = mod(flag - 1, 40);
    filename = sprintf('%sset_%d/Qsave_gdot_%02d.mat', Folder, i, flag-1);
    data = load(filename, 'Qsave');
    xr_tmp = double(data.Qsave(:,:,:,end-100+1:end));
    ff_tmp = get_ff_from_xr(xr_tmp);
    train_rr{flag} = xr_tmp;
    train_ff{flag} = ff_tmp;
    disp([i,j])
end

% Concatenate along first dimension
train_rr = cell2mat(train_rr); % [6   400     3    70]
train_ff = cell2mat(train_ff);
train_stress = squeeze(mean(pagemtimes(permute(train_ff,[3 1 2 4]), permute(train_rr,[1 3 2 4])), 3));

disp(size(train_rr))
train_rr = single(train_rr);
train_ff = single(train_ff);
train_stress = single(train_stress);
save(fullfile(fileparts(mfilename('fullpath')),'sde_bin40_gamma_170.mat'),'train_ff','train_rr','train_stress','-v7.3')


%% -------- Function --------
function ff = get_ff_from_xr(rr)  % [6   400     3    70]
Ks = 0.1;l0 =2.3;l0_sq = l0*l0;

r_bond=vecnorm(rr,2,3); % (6,N,1,70)
rr_r0 = 1.0 - r_bond.*r_bond/l0_sq;
rr_r0(rr_r0<0.01)=0.01;
ff=Ks.*rr./rr_r0; % (6,N,3,70)

end