% DESCRIPTION:
% This script evaluates the predictive performance of gait models 
% using multiple validation strategies:
%
%   1. LOSO (Leave-One-Subject-Out): assesses inter-subject generalization.
%   2. LOCO (Leave-One-Condition-Out): assesses inter-condition generalization.
%
% Each section computes detailed performance metrics, including accuracy,
% correlation, and temporal alignment, and exports results to Excel.
%
% METRICS:
%   - Accuracy: MAE, RMSE, NRMSE, MAPE, sMAPE
%   - Correlation: R², Pearson’s r, Concordance Correlation Coefficient (CCC)
%   - Temporal alignment: Mean lag between predicted and true sequences
%
% AUTHOR:   Azza Tayari
% COPYRIGHT (c) 2025 Azza Tayari. All rights reserved.
% CONTACT:  tayari.azza@eniso.u-sousse.tn
% DATE:     November 2025
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%
% clear; clc;
%% LOSO 
% % === Folder path where all subject CSVs are stored ===
% dataDir = 'D:\codeFrom0\LOSO\';
% files = dir(fullfile(dataDir, 'subject_*_all_predictions.csv'));
% 
% nSubjects = 10;
% 
% % --- Preallocate numeric arrays ---
% MAE_L = zeros(nSubjects,1); RMSE_L = zeros(nSubjects,1); R2_L = zeros(nSubjects,1);
% Lag_L = zeros(nSubjects,1); MAPE_L = zeros(nSubjects,1); sMAPE_L = zeros(nSubjects,1);
% r_L = zeros(nSubjects,1); CCC_L = zeros(nSubjects,1); NRMSE_L = zeros(nSubjects,1);
% 
% MAE_R = zeros(nSubjects,1); RMSE_R = zeros(nSubjects,1); R2_R = zeros(nSubjects,1);
% Lag_R = zeros(nSubjects,1); MAPE_R = zeros(nSubjects,1); sMAPE_R = zeros(nSubjects,1);
% r_R = zeros(nSubjects,1); CCC_R = zeros(nSubjects,1); NRMSE_R = zeros(nSubjects,1);
% 
% Corr_LR_Truth = zeros(nSubjects,1); Corr_LR_Pred = zeros(nSubjects,1);
% Corr_Lpred_Ltruth = zeros(nSubjects,1); Corr_Rpred_Rtruth = zeros(nSubjects,1);
% 
% FileNames = strings(nSubjects,1);
% 
% for i = 1:nSubjects
%     subjName = files(i).name;
%     subjPath = fullfile(dataDir, subjName);
%     data = readmatrix(subjPath);
% 
%     Lpred = data(:,2); Rpred = data(:,3);
%     Ltruth = data(:,4); Rtruth = data(:,5);
% 
%     % --- Compute metrics for Left ---
%     [MAE_L(i), RMSE_L(i), R2_L(i), Lag_L(i), MAPE_L(i), sMAPE_L(i), r_L(i), CCC_L(i), NRMSE_L(i)] = computeMetrics(Ltruth, Lpred);
% 
%     % --- Compute metrics for Right ---
%     [MAE_R(i), RMSE_R(i), R2_R(i), Lag_R(i), MAPE_R(i), sMAPE_R(i), r_R(i), CCC_R(i), NRMSE_R(i)] = computeMetrics(Rtruth, Rpred);
% 
%     % --- Cross-variable correlations ---
%     Corr_LR_Truth(i) = corr(Ltruth, Rtruth, 'Type', 'Pearson');
%     Corr_LR_Pred(i) = corr(Lpred, Rpred, 'Type', 'Pearson');
%     Corr_Lpred_Ltruth(i) = corr(Lpred, Ltruth, 'Type', 'Pearson');
%     Corr_Rpred_Rtruth(i) = corr(Rpred, Rtruth, 'Type', 'Pearson');
% 
%     FileNames(i) = string(subjName);
% end
% 
% % --- Create table at the end (fast!) ---
% Results = table((1:nSubjects)', FileNames, MAE_L, RMSE_L, R2_L, Lag_L, MAPE_L, sMAPE_L, r_L, CCC_L, NRMSE_L, ...
%     MAE_R, RMSE_R, R2_R, Lag_R, MAPE_R, sMAPE_R, r_R, CCC_R, NRMSE_R, ...
%     Corr_LR_Truth, Corr_LR_Pred, Corr_Lpred_Ltruth, Corr_Rpred_Rtruth, ...
%     'VariableNames', { ...
%     'SubjectID', 'File', ...
%     'MAE_L', 'RMSE_L', 'R2_L', 'Lag_L', 'MAPE_L', 'sMAPE_L', 'r_L', 'CCC_L', 'NRMSE_L', ...
%     'MAE_R', 'RMSE_R', 'R2_R', 'Lag_R', 'MAPE_R', 'sMAPE_R', 'r_R', 'CCC_R', 'NRMSE_R', ...
%     'Corr_LR_Truth', 'Corr_LR_Pred', 'Corr_Lpred_Ltruth', 'Corr_Rpred_Rtruth'});
% 
% % --- Compute overall metrics ---
% metricNames = Results.Properties.VariableNames(3:22); % All numeric metrics
% overallMean = zeros(1,length(metricNames));
% overallStd  = zeros(1,length(metricNames));
% 
% for k = 1:length(metricNames)
%     overallMean(k) = mean(Results.(metricNames{k}));
%     overallStd(k)  = std(Results.(metricNames{k}));
% end
% 
% % Display as table
% OverallMetrics = table(metricNames', overallMean', overallStd', ...
%     'VariableNames', {'Metric','Mean','Std'});
% 
% disp('--- Overall Metrics ---');
% disp(OverallMetrics);
% 
% 
% % --- Save results ---
% outFile = fullfile(dataDir, 'LOSO_AllSubjects_Metrics.xlsx');
% writetable(Results, outFile);
% disp(['✅ Results saved to: ' outFile]);
% 
% % --- Metric function ---
% function [mae, rmse, r2, lag_frames, mape, smape, r, ccc, nrmse] = computeMetrics(y, p)
%     e = y - p;
% 
%     mae = mean(abs(e));
%     rmse = sqrt(mean(e.^2));
%     SSres = sum(e.^2);
%     SStot = sum((y - mean(y)).^2);
%     r2 = 1 - SSres / SStot;
% 
%     r = corr(y, p, 'Type', 'Pearson');
% 
%     mean_y = mean(y); mean_p = mean(p);
%     var_y = var(y); var_p = var(p);
%     ccc = (2 * r * sqrt(var_y * var_p)) / (var_y + var_p + (mean_y - mean_p)^2);
% 
%     nrmse = (rmse / (max(y)-min(y))) * 100;
% 
%     [xc,lags] = xcorr(p-mean(p), y-mean(y), 'coeff');
%     [~, idx] = max(xc);
%     lag_frames = lags(idx);
% 
%     epsilon = 1e-6;
%     mape  = mean(abs(e./(y+epsilon))) * 100;
%     smape = mean(2*abs(e)./(abs(y)+abs(p)+epsilon)) * 100;
% end

%% LOCO 

% === Folder path where all trial CSVs are stored === 
dataDir = 'D:\codeFrom0\LOCO\';

% --- Define the specific trials to process ---
trialIDs = [33,31, 30, 29, 28, 27, 26, 15, 13, 12, 10, 9, 7, 6, 4, 3];
nTrials = length(trialIDs);

% --- Preallocate numeric arrays ---
MAE_L = zeros(nTrials,1); RMSE_L = zeros(nTrials,1); R2_L = zeros(nTrials,1);
Lag_L = zeros(nTrials,1); MAPE_L = zeros(nTrials,1); sMAPE_L = zeros(nTrials,1);
r_L = zeros(nTrials,1); CCC_L = zeros(nTrials,1); NRMSE_L = zeros(nTrials,1);

MAE_R = zeros(nTrials,1); RMSE_R = zeros(nTrials,1); R2_R = zeros(nTrials,1);
Lag_R = zeros(nTrials,1); MAPE_R = zeros(nTrials,1); sMAPE_R = zeros(nTrials,1);
r_R = zeros(nTrials,1); CCC_R = zeros(nTrials,1); NRMSE_R = zeros(nTrials,1);

Corr_LR_Truth = zeros(nTrials,1); Corr_LR_Pred = zeros(nTrials,1);
Corr_Lpred_Ltruth = zeros(nTrials,1); Corr_Rpred_Rtruth = zeros(nTrials,1);

FileNames = strings(nTrials,1);

% === Main loop through selected trials ===
for i = 1:nTrials
    trialNum = trialIDs(i);
    fileName = sprintf('trial_%d_all_predictions.csv', trialNum);
    filePath = fullfile(dataDir, fileName);

    if ~isfile(filePath)
        warning('⚠️ File not found: %s', fileName);
        continue;
    end

    data = readmatrix(filePath);

    Lpred = data(:,3); Rpred = data(:,5);
    Ltruth = data(:,2); Rtruth = data(:,4);

    % --- Compute metrics for Left ---
    [MAE_L(i), RMSE_L(i), R2_L(i), Lag_L(i), MAPE_L(i), sMAPE_L(i), r_L(i), CCC_L(i), NRMSE_L(i)] = computeMetrics(Ltruth, Lpred);

    % --- Compute metrics for Right ---
    [MAE_R(i), RMSE_R(i), R2_R(i), Lag_R(i), MAPE_R(i), sMAPE_R(i), r_R(i), CCC_R(i), NRMSE_R(i)] = computeMetrics(Rtruth, Rpred);

    % --- Cross-variable correlations ---
    Corr_LR_Truth(i) = corr(Ltruth, Rtruth, 'Type', 'Pearson');
    Corr_LR_Pred(i) = corr(Lpred, Rpred, 'Type', 'Pearson');
    Corr_Lpred_Ltruth(i) = corr(Lpred, Ltruth, 'Type', 'Pearson');
    Corr_Rpred_Rtruth(i) = corr(Rpred, Rtruth, 'Type', 'Pearson');

    FileNames(i) = string(fileName);
end

% --- Create results table ---
Results = table(trialIDs', FileNames, MAE_L, RMSE_L, R2_L, Lag_L, MAPE_L, sMAPE_L, r_L, CCC_L, NRMSE_L, ...
    MAE_R, RMSE_R, R2_R, Lag_R, MAPE_R, sMAPE_R, r_R, CCC_R, NRMSE_R, ...
    Corr_LR_Truth, Corr_LR_Pred, Corr_Lpred_Ltruth, Corr_Rpred_Rtruth, ...
    'VariableNames', { ...
    'TrialID', 'File', ...
    'MAE_L', 'RMSE_L', 'R2_L', 'Lag_L', 'MAPE_L', 'sMAPE_L', 'r_L', 'CCC_L', 'NRMSE_L', ...
    'MAE_R', 'RMSE_R', 'R2_R', 'Lag_R', 'MAPE_R', 'sMAPE_R', 'r_R', 'CCC_R', 'NRMSE_R', ...
    'Corr_LR_Truth', 'Corr_LR_Pred', 'Corr_Lpred_Ltruth', 'Corr_Rpred_Rtruth'});

% --- Compute overall metrics ---
metricNames = Results.Properties.VariableNames(3:22);
overallMean = zeros(1,length(metricNames));
overallStd  = zeros(1,length(metricNames));

for k = 1:length(metricNames)
    overallMean(k) = mean(Results.(metricNames{k}), 'omitnan');
    overallStd(k)  = std(Results.(metricNames{k}), 'omitnan');
end

% --- Display overall metrics ---
OverallMetrics = table(metricNames', overallMean', overallStd', ...
    'VariableNames', {'Metric','Mean','Std'});
disp('--- Overall Metrics ---');
disp(OverallMetrics);

% --- Save per-trial results to Excel ---
outFile = fullfile(dataDir, 'LOCO_AllSubjects_Metrics.xlsx');
writetable(Results, outFile);
disp(['✅ LOCO results saved to: ' outFile]);

function [mae, rmse, r2, lag_frames, mape, smape, r, ccc, nrmse] = computeMetrics(y, p)
    % --- Ensure valid numeric input ---
    y = y(:); p = p(:);
    validIdx = ~(isnan(y) | isnan(p));
    y = y(validIdx);
    p = p(validIdx);

    if isempty(y)
        mae = NaN; rmse = NaN; r2 = NaN; lag_frames = NaN;
        mape = NaN; smape = NaN; r = NaN; ccc = NaN; nrmse = NaN;
        return;
    end

    % --- Basic error vector ---
    e = y - p;

    % --- Core metrics ---
    mae = mean(abs(e), 'omitnan');
    rmse = sqrt(mean(e.^2, 'omitnan'));

    SSres = sum(e.^2, 'omitnan');
    SStot = sum((y - mean(y, 'omitnan')).^2, 'omitnan');
    r2 = 1 - SSres / max(SStot, eps);  % avoid divide-by-zero

    % --- Pearson correlation ---
    if std(y) == 0 || std(p) == 0
        r = NaN;
    else
        r = corr(y, p, 'Type', 'Pearson', 'Rows', 'complete');
    end

    % --- Concordance Correlation Coefficient (CCC) ---
    mean_y = mean(y, 'omitnan'); mean_p = mean(p, 'omitnan');
    var_y = var(y, 'omitnan'); var_p = var(p, 'omitnan');
    ccc = (2 * r * sqrt(var_y * var_p)) / (var_y + var_p + (mean_y - mean_p)^2 + eps);

    % --- Normalized RMSE ---
    rangeY = max(y) - min(y);
    if rangeY > 0
        nrmse = (rmse / rangeY) * 100;
    else
        nrmse = NaN;
    end

    % --- Cross-correlation lag (robust) ---
    try
        [xc, lags] = xcorr(p - mean(p), y - mean(y), 'coeff');
        [~, idx] = max(xc);
        lag_frames = lags(idx);
    catch
        lag_frames = NaN;
    end

    % --- MAPE and sMAPE (robust) ---
    epsilon = 1e-6;
    denom = abs(y); denom(denom < epsilon) = epsilon;
    mape = mean(abs(e) ./ denom) * 100;
    smape = mean(2 * abs(e) ./ (abs(y) + abs(p) + epsilon)) * 100;
end