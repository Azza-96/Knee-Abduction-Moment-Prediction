
% Description:
% This script evaluates the performance of predicted Knee Adduction Moment
% (KAM) signals against the original ground-truth data for each subject and
% gait trial. The analysis quantifies waveform agreement using a set of 
% time-domain and correlation-based metrics.
%
% -------------------------------------------------------------------------
% Structure Overview:
% 1) Load and preprocess the original and predicted KAM datasets
%    - Reads ground-truth and model-predicted KAM signals (left/right)
%    - Downsamples and aligns data for comparison
%
% 2) Extract trial and subject identifiers from the file headers
%    - Automatically parses "subjID" and "trialID" for each dataset column
%
% 3) Compute evaluation metrics for each subject-trial pair
%    - Includes MAE, RMSE, R², Pearson correlation, CCC, NRMSE,
%      lag (cross-correlation), DTW distance, MAPE, and sMAPE
%
% 4) Summarize results across all trials and subjects
%    - Outputs detailed metrics per trial and summary statistics
%
% 5) Visualization
%    - Plots example comparison between true and predicted KAM waveforms
%      for selected subject and trial (left/right legs)
%
% -------------------------------------------------------------------------
% Supporting Functions:
% - computeMetrics(): Calculates waveform error and similarity metrics
% - getTrialAbbreviation(): Returns short descriptive label for each trial
%
% -------------------------------------------------------------------------
% Copyright © 2025 Azza Tayari

close all; clear all; clc;

%% 1) Load and downsample original KAM data
left_KAM  = downsample(readmatrix('D:\codeFrom0\ExtractedCSV\Validation\left_knee_moment_axis2_validation.csv'), 10);
right_KAM = downsample(readmatrix('D:\codeFrom0\ExtractedCSV\Validation\right_knee_moment_axis2_validation.csv'), 10);
left_KAM  = left_KAM(100:end, :);
right_KAM = right_KAM(100:end, :);

% Load prediction files
predFileL = 'D:\codeFrom0\all_trials_predicted_left_columns.csv';
predFileR = 'D:\codeFrom0\all_trials_predicted_right_columns.csv';
left_Pred  = readmatrix(predFileL);
right_Pred = readmatrix(predFileR);

assert(isequal(size(left_KAM),  size(left_Pred)),  'Left size mismatch');
assert(isequal(size(right_KAM), size(right_Pred)), 'Right size mismatch');

%% 2) Extract subjID & trialID from headers
opts  = detectImportOptions(predFileL);
vars  = opts.VariableNames;
Ncol  = numel(vars);
subjID  = zeros(1, Ncol);
trialID = zeros(1, Ncol);
for k = 1:Ncol
    tok = regexp(vars{k}, 'subj(\d+)_trial(\d+)', 'tokens', 'once');
    subjID(k)  = str2double(tok{1});
    trialID(k) = str2double(tok{2});
end
subjects = unique(subjID);

%% 3) Process trials and compute metrics
results = [];
for s = subjects
    idxSubj = find(subjID == s);
    trialsForSubj = unique(trialID(idxSubj));
    
    for t = trialsForSubj
        if t == 19
            continue; % Skip trial 19
        end
        
        col = idxSubj(trialID(idxSubj) == t);
        
        yL = left_KAM(:, col);
        yR = right_KAM(:, col);
        pL = left_Pred(:, col);
        pR = right_Pred(:, col);
        
% --- Compute metrics for left and right knee predictions ---
[maeL, rmseL, r2L, lagL, dtwL, mapeL, smapeL, rL, cccL, nrmseL] = computeMetrics(yL, pL);
[maeR, rmseR, r2R, lagR, dtwR, mapeR, smapeR, rR, cccR, nrmseR] = computeMetrics(yR, pR);

% --- Store results in structured format for table conversion ---
rec = struct( ...
    'Subject', s, ...
    'Trial', t, ...
    'MAE_L', maeL, 'RMSE_L', rmseL, 'R2_L', r2L, ...
    'r_L', rL, 'CCC_L', cccL, 'NRMSE_L', nrmseL, ...
    'MAE_R', maeR, 'RMSE_R', rmseR, 'R2_R', r2R, ...
    'r_R', rR, 'CCC_R', cccR, 'NRMSE_R', nrmseR, ...
    'MAPE_L', mapeL, 'MAPE_R', mapeR, ...       % ✅ Now included
    'sMAPE_L', smapeL, 'sMAPE_R', smapeR, ...   % ✅ Symmetric MAPE
    'Lag_L', lagL, 'Lag_R', lagR, ...
    'DTW_L', dtwL, 'DTW_R', dtwR ...
);

        results = [results; rec]; %#ok<AGROW>
    end
end

% Convert results to table
T = struct2table(results);

% Add trial abbreviation column
T.TrialAbbr = strings(height(T), 1);
for i = 1:height(T)
    T.TrialAbbr(i) = getTrialAbbreviation(T.Trial(i));
end

%% 4) Summary Statistics (Updated with waveform agreement metrics)
fprintf('\n--- Summary Metrics ---\n');
fprintf('Left  MAE:    %.3f ± %.3f Nm\n', mean(T.MAE_L),  std(T.MAE_L));
fprintf('Right MAE:    %.3f ± %.3f Nm\n', mean(T.MAE_R),  std(T.MAE_R));
fprintf('Left  RMSE:   %.3f ± %.3f Nm\n', mean(T.RMSE_L), std(T.RMSE_L));
fprintf('Right RMSE:   %.3f ± %.3f Nm\n', mean(T.RMSE_R), std(T.RMSE_R));
fprintf('Left  NRMSE:  %.3f ± %.3f\n',   mean(T.NRMSE_L), std(T.NRMSE_L));
fprintf('Right NRMSE:  %.3f ± %.3f\n',   mean(T.NRMSE_R), std(T.NRMSE_R));
fprintf('Left  R²:     %.3f ± %.3f\n',   mean(T.R2_L),    std(T.R2_L));
fprintf('Right R²:     %.3f ± %.3f\n',   mean(T.R2_R),    std(T.R2_R));
fprintf('Left  r:      %.3f ± %.3f\n',   mean(T.r_L),     std(T.r_L));
fprintf('Right r:      %.3f ± %.3f\n',   mean(T.r_R),     std(T.r_R));
fprintf('Left  CCC:    %.3f ± %.3f\n',   mean(T.CCC_L),   std(T.CCC_L));
fprintf('Right CCC:    %.3f ± %.3f\n',   mean(T.CCC_R),   std(T.CCC_R));
fprintf('Mean Lag Left:   %.2f frames\n', mean(T.Lag_L));
fprintf('Mean Lag Right:  %.2f frames\n', mean(T.Lag_R));
fprintf('Mean DTW Left:   %.2f\n',        mean(T.DTW_L));
fprintf('Mean DTW Right:  %.2f\n',        mean(T.DTW_R)); 

%% 5) Optional Plot Example
subj2plot   = subjects(10);
trial2plot  = unique(trialID(12));
trial2plot  = trial2plot(trial2plot ~= 19); % Exclude trial 19
trial2plot  = trial2plot(1);
col = find(subjID == subj2plot & trialID == trial2plot);

%% Create subplot figure
figure;

% --- Left KAM subplot ---
subplot(2, 1, 1);
plot(left_KAM(:, col),  'g', 'LineWidth', 1.5, 'DisplayName', 'True Left KAM'); hold on;
plot(left_Pred(:, col), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Left KAM');
legend('Location', 'best');
grid on;set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');
xlabel('Frame');
ylabel('KAM (Nm/kg)');
title(sprintf('Left KAM\nSubject %d, Trial %d', subj2plot, trial2plot), 'FontWeight', 'bold');

% --- Right KAM subplot ---
subplot(2,1, 2);
plot(right_KAM(:, col),  'r', 'LineWidth', 1.5, 'DisplayName', 'True Right KAM'); hold on;
plot(right_Pred(:, col), 'm--', 'LineWidth', 1.5, 'DisplayName', 'Predicted Right KAM');
legend('Location', 'best');
grid on;
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');
xlabel('Frame');
ylabel('KAM(Nm/kg)');
title(sprintf('Right KAM\nSubject %d, Trial %d', subj2plot, trial2plot), 'FontWeight', 'bold');

% Optional: Improve overall layout
sgtitle('Knee Abduction Moment: True vs Predicted', 'FontSize', 14, 'FontWeight', 'bold');


function [mae, rmse, r2, lag_frames, dtw_dist, mape, smape, r, ccc, nrmse] = computeMetrics(y, p)
    e = y - p;

    % --- Basic Errors ---
    mae = mean(abs(e));
    rmse = sqrt(mean(e.^2));

    % --- R² ---
    SSres = sum(e.^2);
    SStot = sum((y - mean(y)).^2);
    r2 = 1 - SSres / SStot;

    % --- Pearson correlation ---
    r = corr(y, p, 'Type', 'Pearson');

    % --- Concordance Correlation Coefficient (CCC) ---
    mean_y = mean(y); mean_p = mean(p);
    var_y = var(y); var_p = var(p);
    ccc = (2 * r * sqrt(var_y * var_p)) / (var_y + var_p + (mean_y - mean_p)^2);

    % --- Normalized RMSE (% peak-to-peak) ---
    nrmse = (rmse / (max(y) - min(y))) * 100;

    % --- Lag ---
    [xc, lags] = xcorr(p - mean(p), y - mean(y), 'coeff');
    [~, idx] = max(xc);
    lag_frames = lags(idx);

    % --- DTW Distance ---
    dtw_dist = dtw(p, y);

    % --- Percentage Errors ---
    epsilon = 1e-6;
    mape  = mean(abs(e ./ (y + epsilon))) * 100;
    smape = mean(2 * abs(e) ./ (abs(y) + abs(p) + epsilon)) * 100;
end
%%

function abbreviation = getTrialAbbreviation(trialNumber)
    switch trialNumber
        case 1, abbreviation = 'VSFr';
        case 2, abbreviation = 'VS';
        case 3, abbreviation = 'VSSL';
        case 4, abbreviation = 'SFr';
        case 5, abbreviation = 'S';
        case 6, abbreviation = 'SSL';
        case 7, abbreviation = 'MFr';
        case 8, abbreviation = 'LN';
        case 9, abbreviation = 'PSL';
        case 10, abbreviation = 'LSL';
        case 11, abbreviation = 'F';
        case 12, abbreviation = 'VHFr';
        case 13, abbreviation = 'VLSL';
        case 14, abbreviation = 'M_F_r';
        case 15, abbreviation = 'H_F_r';
        case 16, abbreviation = 'VF';
        case 26, abbreviation = 'VSSW';
        case 27, abbreviation = 'SSW';
        case 28, abbreviation = 'MSW';
        case 29, abbreviation = 'WSW';
        case 30, abbreviation = 'VWSW';
        case 31, abbreviation = 'HFr';
        case 32, abbreviation = 'VS';
        case 33, abbreviation = 'ISL';
        otherwise, abbreviation = '';
    end
end

