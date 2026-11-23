
clear; clc; close all;

%% ---------------- USER SETTINGS ----------------
simRoot = 'D:\ANTARIKSH\MLWC_3\Assignment 2 MLWC supporting files\5GLLSimulator\5GLLSimulator';
dataDir = 'D:\ANTARIKSH\MLWC_3\data';      % output folder (created if missing). Keep it OUT of OneDrive.
% exportList = { 'VehA', 12, 48, 1,     5;   % channel, SNR (dB), pilots, first sample, count
%                'SUI5', 22, 48, 36001, 5 };

exportList = { 'VehA', 12, 48, 1,     200;
    'VehA', 22, 48, 36001, 200;
    'SUI5', 12, 48, 1,     200;
    'SUI5', 22, 48, 36001, 200 };

%% ---------------- SETUP ----------------
if exist(fullfile(simRoot,'+Modulation'),'dir') ~= 7
    error('simRoot is wrong. The folder must directly contain +Modulation and +Channel:\n%s', simRoot);
end
addpath(simRoot);
if ~exist(dataDir,'dir'), mkdir(dataDir); end
cfg      = a3_config();
chans    = {'VehA','SUI5'};
snrAll   = [0 5 10 12 15 20 22 25 30];
spacings = [18 12 9 6];                    % 16, 24, 32, 48 pilots (6 = main case)
methods  = {'linear','natural'};

%% ---------------- STAGE 1: ground-truth channels (Step 3) ----------------
fprintf('\n== Stage 1: channels ==\n');
for c = 1:2
    f = fullfile(dataDir, ['H_' chans{c} '.mat']);
    if exist(f,'file')
        fprintf('[skip] %s already exists\n', f);
    else
        tic;  a3_gen_channels(cfg, chans{c}, dataDir);
        fprintf('  done in %.1f s\n', toc);
    end
end

%% ---------------- STAGE 2: noisy LS estimates at the pilots (Step 5) ----------------
fprintf('\n== Stage 2: LS estimates ==\n');
for c = 1:2
    S = load(fullfile(dataDir, ['H_' chans{c} '.mat']), 'Hre', 'Him');
    for snr = snrAll
        a3_gen_ls(S.Hre, S.Him, cfg, chans{c}, 6, snr, dataDir);
    end
    for sp = [18 12 9]
        a3_gen_ls(S.Hre, S.Him, cfg, chans{c}, sp, 22, dataDir);
    end
end
clear S;

%% ---------------- STAGE 3: interpolation + LS baseline (Step 6) ----------------
fprintf('\n== Stage 3: LS + 2D interpolation baseline ==\n');
for s = 1:numel(spacings)
    [~, pIdx] = a3_pilot_mask(cfg, spacings(s));
    for m = 1:numel(methods)
        W = a3_interp_weights(cfg, pIdx, methods{m});
        Wint(s,m).mat  = W;
        Wint(s,m).pIdx = pIdx;
        save(fullfile(dataDir, sprintf('W_%s_P%d.mat', methods{m}, numel(pIdx))), 'W', '-v7');  % plain matrix for Python
    end
end
save(fullfile(dataDir,'W_interp.mat'), 'Wint', 'spacings', 'methods');
Wt = Wint(4,1).mat;  pI = Wint(4,1).pIdx;
fprintf('W check: pass-through error %.1e, row-sum error %.1e (both should be ~0)\n', ...
        norm(Wt(pI,:) - eye(numel(pI)), 'fro'), max(abs(sum(Wt,2) - 1)));

mseSNR = zeros(2, numel(methods), numel(snrAll));    % channel x method x SNR (48 pilots)
msePil = zeros(2, numel(methods), numel(spacings));  % channel x method x pilot count (22 dB)
for c = 1:2
    S  = load(fullfile(dataDir, ['H_' chans{c} '.mat']), 'Hre', 'Him', 'idx');
    ti = S.idx.test;
    for m = 1:numel(methods)
        for k = 1:numel(snrAll)
            L = load(fullfile(dataDir, sprintf('LS_%s_P48_snr%d.mat', chans{c}, snrAll(k))), 'HlsRe', 'HlsIm');
            mseSNR(c,m,k) = a3_mse(Wint(4,m).mat, L.HlsRe, L.HlsIm, S.Hre, S.Him, ti);
        end
        for s = 1:numel(spacings)
            nP = numel(Wint(s,m).pIdx);
            L  = load(fullfile(dataDir, sprintf('LS_%s_P%d_snr22.mat', chans{c}, nP)), 'HlsRe', 'HlsIm');
            msePil(c,m,s) = a3_mse(Wint(s,m).mat, L.HlsRe, L.HlsIm, S.Hre, S.Him, ti);
        end
    end
end
clear S L;
save(fullfile(dataDir,'baseline_LS_interp.mat'), 'mseSNR', 'msePil', 'snrAll', 'spacings', 'methods', 'chans');

for c = 1:2
    fprintf('\n%s, linear interpolation, MSE on the test split\n', chans{c});
    fprintf('  SNR (dB)          : %s\n', sprintf('%8d', snrAll));
    fprintf('  48 pilots         : %s\n', sprintf('%8.4f', squeeze(mseSNR(c,1,:))));
    fprintf('  16/24/32/48 @22dB : %s\n', sprintf('%8.4f', squeeze(msePil(c,1,:))));
end

figure; hold on;
sty = {'-o','--o'; '-s','--s'};
for c = 1:2
    for m = 1:numel(methods)
        semilogy(snrAll, squeeze(mseSNR(c,m,:)), sty{c,m}, 'DisplayName', [chans{c} ' ' methods{m}]);
    end
end
set(gca,'YScale','log'); grid on; xlabel('SNR (dB)'); ylabel('MSE'); legend('Location','southwest');
title('LS + 2D interpolation, 48 pilots');

%% ---------------- STAGE 4: export tables ----------------
fprintf('\n== Stage 4: tables ==\n');
for e = 1:size(exportList,1)
    outBase = fullfile(dataDir, 'tables', sprintf('%s_snr%d_P%d_from%d', ...
              exportList{e,1}, exportList{e,2}, exportList{e,3}, exportList{e,4}));
    a3_export(cfg, dataDir, exportList{e,1}, exportList{e,2}, exportList{e,3}, ...
              exportList{e,4}, exportList{e,5}, outBase);
end
fprintf('\nALL DONE. Files are in %s\n', dataDir);

%% ======================= LOCAL FUNCTIONS =======================

function cfg = a3_config()
cfg.Nsc = 72;  cfg.Nsym = 14;
cfg.scs = 15e3;  cfg.nfft = 128;  cfg.fs = cfg.scs*cfg.nfft;   % 1.92 MHz
cfg.cpSamples = 9;                                             % LTE normal CP
cfg.fc = 2.1e9;  cfg.v_kmh = 50;
cfg.fd = cfg.v_kmh/3.6*cfg.fc/2.998e8;                         % about 97.3 Hz
cfg.nPaths = 200;                                              % Jakes paths
cfg.tSub = 137;                                                % channel time-subsampling = samples per OFDM symbol
cfg.nTrain = 32000;  cfg.nVal = 4000;  cfg.nTest = 4000;
cfg.seed = 42;
end

function [ch, tp, nfft] = a3_channel(cfg, chanName)
ofdm = Modulation.OFDM(cfg.Nsc, cfg.Nsym, cfg.scs, cfg.fs, 0, false, cfg.cpSamples/cfg.fs, 0);
N    = ofdm.Nr.SamplesTotal;
nfft = ofdm.GetFFTsize;
tp   = ofdm.GetTimeIndexMidPos;
switch chanName
    case 'VehA'
        pdp = 'VehicularA';
    case 'SUI5'
        pdp = zeros(1,20);  pdp([1 9 20]) = 10.^([0 -5 -10]/10);   % taps at 0, 4, 10 us
    otherwise
        error('Unknown channel: %s', chanName);
end
dummy.spatialChannelModel = false;
ch = Channel.FastFading(cfg.fs, pdp, N, cfg.fd, 'Jakes', cfg.nPaths, false, 'none', ...
        0, 0, 1, 1, false, 0, 1, dummy, [1,1], [1,1], cfg.fc, cfg.tSub);
end

function a3_gen_channels(cfg, chanName, dataDir)
nTot = cfg.nTrain + cfg.nVal + cfg.nTest;
seed = cfg.seed + strcmp(chanName,'SUI5');     % 42 for VehA, 43 for SUI5
rng(seed);
[ch, tp, nfft] = a3_channel(cfg, chanName);
Hre = zeros(cfg.Nsc, cfg.Nsym, nTot, 'single');
Him = zeros(cfg.Nsc, cfg.Nsym, nTot, 'single');
for n = 1:nTot
    ch.NewRealization;
    H = squeeze(ch.GetTransferFunction(tp, nfft, 1:cfg.Nsc));   % Nsc x Nsym complex
    Hre(:,:,n) = real(H);
    Him(:,:,n) = imag(H);
end
idx.train = 1:cfg.nTrain;
idx.val   = cfg.nTrain + (1:cfg.nVal);
idx.test  = cfg.nTrain + cfg.nVal + (1:cfg.nTest);
save(fullfile(dataDir, ['H_' chanName '.mat']), 'Hre', 'Him', 'idx', 'cfg', 'seed', '-v7.3');

% quick checks: size, power, and the slot-1 vs slot-14 correlation against Jakes J0
P   = mean(Hre.^2 + Him.^2, 'all');
H1  = complex(Hre(:,1,1:2000),        Him(:,1,1:2000));
H14 = complex(Hre(:,cfg.Nsym,1:2000), Him(:,cfg.Nsym,1:2000));
rho = real(mean(H1 .* conj(H14), 'all') / mean(abs(H1).^2, 'all'));
tau = (cfg.Nsym-1)*cfg.tSub/cfg.fs;
fprintf('  %s: size %s, seed %d, power %.4f, lag-13 corr %.4f (J0 theory %.4f)\n', ...
        chanName, mat2str(size(Hre)), seed, P, rho, besselj(0, 2*pi*cfg.fd*tau));
end

function [mask, pIdx] = a3_pilot_mask(cfg, spacing)
% Lattice-type pilot mask on the Nsc x Nsym grid (true = pilot).
tsyms = [1 5 8 12];                              % pilot OFDM symbols (1-based)
mask  = false(cfg.Nsc, cfg.Nsym);
for i = 1:numel(tsyms)
    off = (mod(i,2)==0) * floor(spacing/2);      % even-numbered pilot symbols are staggered
    mask(1+off : spacing : cfg.Nsc, tsyms(i)) = true;
end
pIdx = find(mask);                               % linear (column-major) indices
end

function a3_gen_ls(Hre, Him, cfg, chanName, spacing, snrdB, outDir)
% Pilot symbols + noise + LS estimate at the pilot positions.
N = size(Hre, 3);
[~, pIdx] = a3_pilot_mask(cfg, spacing);
nP = numel(pIdx);
Hre2 = reshape(Hre, [], N);  Him2 = reshape(Him, [], N);
Hp = double(Hre2(pIdx, :)) + 1j*double(Him2(pIdx, :));          % nP x N true channel at pilots

seedP = 2024;
rsP = RandStream('mt19937ar', 'Seed', seedP);
Xp  = exp(1j*(pi/4 + (pi/2)*randi(rsP, [0 3], nP, 1)));          % unit-power QPSK pilots

sigma2 = 10^(-snrdB/10);                                         % E|H|^2 = 1, |X|^2 = 1
c = find(strcmp(chanName, {'VehA','SUI5'}));
seedN = 1e5*c + 1e3*spacing + snrdB;
rsN = RandStream('mt19937ar', 'Seed', seedN);
Z = sqrt(sigma2/2) * (randn(rsN, nP, N) + 1j*randn(rsN, nP, N));

Y   = Hp .* Xp + Z;                                              % Eq. (1) at the pilots
Hls = Y ./ Xp;                                                   % Eq. (2) LS estimate
lsErr = mean(abs(Hls - Hp).^2, 'all');
fprintf('  %s P=%2d SNR=%2d dB: LS error at pilots = %.5f (theory %.5f)\n', chanName, nP, snrdB, lsErr, sigma2);

HlsRe = single(real(Hls));  HlsIm = single(imag(Hls));
save(fullfile(outDir, sprintf('LS_%s_P%d_snr%d.mat', chanName, nP, snrdB)), ...
     'HlsRe', 'HlsIm', 'pIdx', 'Xp', 'snrdB', 'spacing', 'nP', 'seedP', 'seedN', 'lsErr');
end

function W = a3_interp_weights(cfg, pIdx, method)
% Interpolation matrix W (Ngrid x nP): full-grid estimate = W * (LS values at pilots).
[sc, sl] = ndgrid(1:cfg.Nsc, 1:cfg.Nsym);       % sc = subcarrier index, sl = slot index
xp = sl(pIdx);  yp = sc(pIdx);                  % x = time, y = subcarrier
nP = numel(pIdx);
F  = scatteredInterpolant(xp, yp, zeros(nP,1), method, 'nearest');
W  = zeros(numel(sc), nP);
for p = 1:nP
    v = zeros(nP,1);  v(p) = 1;
    F.Values = v;
    W(:,p) = F(sl(:), sc(:));
end
end

function e = a3_mse(Wt, LSre, LSim, Hre, Him, ti)
% MSE of the interpolated LS estimate over the full grid, on the samples in ti.
Hr = reshape(double(Hre(:,:,ti)), [], numel(ti));
Hi = reshape(double(Him(:,:,ti)), [], numel(ti));
Er = Wt*double(LSre(:,ti)) - Hr;
Ei = Wt*double(LSim(:,ti)) - Hi;
e  = mean(Er(:).^2 + Ei(:).^2);
end

function a3_export(cfg, dataDir, chanName, snrdB, nP, first, count, outBase)
% Export samples first : first+count-1 to three CSV tables (grid, pilots, summary).
samples = first : first+count-1;
if samples(end) > cfg.nTrain + cfg.nVal + cfg.nTest
    error('Sample range exceeds the dataset size.');
end
mf  = matfile(fullfile(dataDir, ['H_' chanName '.mat']));
Hre = double(mf.Hre(:,:,samples));
Him = double(mf.Him(:,:,samples));

L = load(fullfile(dataDir, sprintf('LS_%s_P%d_snr%d.mat', chanName, nP, snrdB)));
pIdx = L.pIdx(:);   Xp = L.Xp(:);
LSre = double(L.HlsRe(:, samples));   LSim = double(L.HlsIm(:, samples));

spacing = 4*cfg.Nsc/nP;                          % 16->18, 24->12, 32->9, 48->6
[~, pChk] = a3_pilot_mask(cfg, spacing);
assert(isequal(pIdx, pChk(:)), 'Pilot indices in the LS file do not match the pilot mask.');
W = a3_interp_weights(cfg, pIdx, 'linear');
Re_Interp = W * LSre;
Im_Interp = W * LSim;

nG = cfg.Nsc*cfg.Nsym;   cnt = count;   nPl = numel(pIdx);
[sc, sl] = ndgrid(1:cfg.Nsc, 1:cfg.Nsym);
isP = false(nG,1);   isP(pIdx) = true;
Hre2 = reshape(Hre, nG, cnt);   Him2 = reshape(Him, nG, cnt);
Re_LS = nan(nG, cnt);   Im_LS = nan(nG, cnt);
Re_LS(pIdx,:) = LSre;   Im_LS(pIdx,:) = LSim;

Tgrid = table(repelem(samples(:), nG), repmat(sc(:), cnt, 1), repmat(sl(:), cnt, 1), ...
    Hre2(:), Him2(:), repmat(isP, cnt, 1), Re_LS(:), Im_LS(:), Re_Interp(:), Im_Interp(:), ...
    'VariableNames', {'Sample','Subcarrier','Slot','Re_H','Im_H','IsPilot', ...
                      'Re_LS','Im_LS','Re_Interp','Im_Interp'});

Tpil = table(repelem(samples(:), nPl), repmat((1:nPl)', cnt, 1), ...
    repmat(sc(pIdx), cnt, 1), repmat(sl(pIdx), cnt, 1), ...
    reshape(Hre2(pIdx,:), [], 1), reshape(Him2(pIdx,:), [], 1), LSre(:), LSim(:), ...
    repmat(real(Xp), cnt, 1), repmat(imag(Xp), cnt, 1), ...
    'VariableNames', {'Sample','Pilot','Subcarrier','Slot','Re_H','Im_H', ...
                      'Re_LS','Im_LS','Re_Xp','Im_Xp'});

Tsum = table(samples(:), mean(Hre2.^2 + Him2.^2, 1)', ...
    mean((Re_Interp - Hre2).^2 + (Im_Interp - Him2).^2, 1)', ...
    'VariableNames', {'Sample','Mean_Power','LS_Interp_MSE'});

d = fileparts(outBase);
if ~isempty(d) && ~exist(d, 'dir'), mkdir(d); end
writetable(Tgrid, [outBase '_grid.csv']);
writetable(Tpil,  [outBase '_pilots.csv']);
writetable(Tsum,  [outBase '_summary.csv']);
fprintf('  %s, %d pilots, %d dB, samples %d-%d -> %s_*.csv (mean power %.4f, mean LS+interp MSE %.5f)\n', ...
        chanName, nP, snrdB, samples(1), samples(end), outBase, mean(Tsum.Mean_Power), mean(Tsum.LS_Interp_MSE));
end