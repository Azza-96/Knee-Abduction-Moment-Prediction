
%  Description:
%  -------------------------------------------------------------------------
%  This MATLAB script performs **statistical comparison** between the
%  **predicted** and **ground-truth Knee Adduction Moment (KAM) peak values**
%  for each gait trial and condition.
%
%  It evaluates the agreement between model predictions and real data using:
%     • Paired t-tests for mean difference
%     • Holm–Bonferroni correction for multiple comparisons
%     • Cohen’s d for effect size estimation
%     • 95% Confidence Intervals (CI) for effect size
%
%  -------------------------------------------------------------------------
%  Workflow:
%  -------------------------------------------------------------------------
%  1. Define gait trials and load the mean ± SD values for each KAM peak:
%       - LP1: First peak (Left leg)
%       - LP2: Second peak (Left leg)
%       - RP1: First peak (Right leg)
%       - RP2: Second peak (Right leg)
%
%  2. Simulate subject-level data (n = 10 subjects per trial)
%     based on the provided group means and standard deviations.
%
%  3. Perform paired t-tests (Real vs Predicted) for each trial and KAM peak.
%
%  4. Apply Holm–Bonferroni correction to control for multiple testing.
%
%  5. Compute Cohen’s d and corresponding 95% CI for effect size estimation.
%
%  6. Export all results into a formatted Excel table:
%       → D:\codeFrom0\statPeak\Table3_Statistics_Revised.xlsx
%
%  7. Print a detailed summary in the MATLAB console.
%
%  -------------------------------------------------------------------------
%  Outputs:
%  -------------------------------------------------------------------------
%  - Table3_Statistics_Revised.xlsx
%       Columns: Peak | Trial | Real_Mean | Real_SD | Pred_Mean | Pred_SD |
%                 p_raw | p_adj | Cohen's d | CI_low | CI_high
%
%  - Console summary showing adjusted p-values and effect sizes.
%
%  -------------------------------------------------------------------------
%  Notes:
%  -------------------------------------------------------------------------
%  • Data in this script are representative (mean ± SD) and simulate the
%    distribution observed across subjects.
%  • For reproducibility, a fixed random seed (rng(42)) is used.
%  • Holm–Bonferroni ensures family-wise error rate control across trials.
%
%  -------------------------------------------------------------------------
%  Dependencies:
%  -------------------------------------------------------------------------
%  - MATLAB R2021a or later
%  - Statistics and Machine Learning Toolbox
%
%  -------------------------------------------------------------------------
%  © 2025 Azza Tayari. All rights reserved.
%% === Setup ==============================================================
clc; clear; close all;

% Trial names
trials = {'VF','VS','S','LN','F','MFr','VSSL','SSL','PSL','LSL','VLSL',...
          'SSW','MSW','WSW','VWSW','Normal'};
nTrials   = numel(trials);
nSubjects = 10;

%% === Input Data (Mean ± SD) ============================================
% --- REAL ---
real.LP1.mean = [0.50,0.59,0.62,0.63,0.47,0.53,0.60,0.64,0.47,0.52,0.49,0.49,0.54,0.42,0.52,0.53];
real.LP1.std  = [0.20,0.21,0.18,0.24,0.16,0.16,0.19,0.17,0.21,0.19,0.14,0.13,0.18,0.15,0.15,0.23];
real.LP2.mean = [0.36,0.41,0.42,0.36,0.42,0.41,0.41,0.41,0.38,0.42,0.44,0.38,0.39,0.39,0.40,0.39];
real.LP2.std  = [0.15,0.16,0.16,0.17,0.11,0.13,0.16,0.15,0.15,0.11,0.16,0.13,0.14,0.13,0.15,0.17];
real.RP1.mean = [0.44,0.56,0.63,0.58,0.45,0.47,0.52,0.59,0.49,0.50,0.43,0.49,0.51,0.41,0.43,0.43];
real.RP1.std  = [0.18,0.19,0.24,0.26,0.14,0.13,0.19,0.22,0.18,0.19,0.18,0.10,0.18,0.16,0.09,0.17];
real.RP2.mean = [0.27,0.37,0.38,0.31,0.35,0.27,0.35,0.34,0.33,0.35,0.33,0.33,0.30,0.35,0.31,0.28];
real.RP2.std  = [0.16,0.20,0.21,0.23,0.16,0.12,0.19,0.22,0.19,0.14,0.18,0.16,0.19,0.13,0.14,0.15];

% --- PREDICTED ---
pred.LP1.mean = [0.44,0.51,0.47,0.44,0.44,0.43,0.45,0.44,0.43,0.45,0.46,0.45,0.42,0.38,0.46,0.48];
pred.LP1.std  = [0.10,0.13,0.12,0.18,0.12,0.15,0.10,0.12,0.13,0.12,0.17,0.07,0.15,0.15,0.07,0.16];
pred.LP2.mean = [0.31,0.31,0.28,0.25,0.32,0.36,0.29,0.28,0.39,0.37,0.36,0.35,0.29,0.37,0.35,0.33];
pred.LP2.std  = [0.09,0.09,0.13,0.13,0.13,0.08,0.10,0.12,0.11,0.12,0.18,0.11,0.10,0.09,0.10,0.13];
pred.RP1.mean = [0.38,0.46,0.50,0.43,0.41,0.38,0.45,0.46,0.43,0.42,0.38,0.40,0.39,0.38,0.39,0.35];
pred.RP1.std  = [0.11,0.11,0.14,0.17,0.09,0.10,0.12,0.14,0.12,0.11,0.11,0.10,0.14,0.14,0.10,0.13];
pred.RP2.mean = [0.25,0.32,0.34,0.21,0.29,0.29,0.26,0.25,0.34,0.32,0.29,0.31,0.24,0.32,0.29,0.27];
pred.RP2.std  = [0.14,0.22,0.18,0.16,0.14,0.12,0.14,0.14,0.19,0.15,0.16,0.15,0.15,0.14,0.13,0.13];

%% === Simulate Subject-Wise Data =========================================
peakNames = {'LP1','LP2','RP1','RP2'};
nPeaks    = numel(peakNames);
rng(42);

for p = 1:nPeaks
    pk = peakNames{p};
    for i = 1:nTrials
        realData.(pk)(:,i) = normrnd(real.(pk).mean(i), real.(pk).std(i), [nSubjects,1]);
        predData.(pk)(:,i) = normrnd(pred.(pk).mean(i), pred.(pk).std(i), [nSubjects,1]);
    end
end

%% === Statistics: Paired t-test + Holm-Bonferroni + Cohen's d + CI =======
results = struct();
for p = 1:nPeaks
    pk = peakNames{p};
    p_raw = nan(1,nTrials);
    d_val = nan(1,nTrials);
    d_ciLow = nan(1,nTrials);
    d_ciHi  = nan(1,nTrials);

    for i = 1:nTrials
        [~, p_raw(i)] = ttest(realData.(pk)(:,i), predData.(pk)(:,i));
        diff = realData.(pk)(:,i) - predData.(pk)(:,i);
        mD = mean(diff); sD = std(diff);
        d_val(i) = mD / sD;
        t_crit = tinv(0.975, nSubjects-1);
        se_d = sqrt( (1/nSubjects) + (d_val(i)^2)/(2*(nSubjects-1)) );
        d_ciLow(i) = d_val(i) - t_crit*se_d;
        d_ciHi(i)  = d_val(i) + t_crit*se_d;
    end

    % Holm-Bonferroni
    [pSort, sortIdx] = sort(p_raw);
    m = nTrials;
    p_adj = zeros(1,m);
    for k = 1:m
        p_adj(sortIdx(k)) = min( (m - k + 1) * pSort(k), 1 );
    end

    results.(pk).p_raw    = p_raw;
    results.(pk).p_adj    = p_adj;
    results.(pk).cohen_d  = d_val;
    results.(pk).CI_low   = d_ciLow;
    results.(pk).CI_high  = d_ciHi;
end


%% === Export Table 3 =====================================================
T = table();
for p = 1:nPeaks
    pk = peakNames{p};
    tmp = table(repmat({pk},nTrials,1), trials', ...
                real.(pk).mean', real.(pk).std', ...
                pred.(pk).mean', pred.(pk).std', ...
                results.(pk).p_raw', results.(pk).p_adj', ...
                results.(pk).cohen_d', results.(pk).CI_low', results.(pk).CI_high', ...
                'VariableNames',{'Peak','Trial','Real_Mean','Real_SD','Pred_Mean','Pred_SD',...
                                 'p_raw','p_adj','Cohens_d','CI_low','CI_high'});
    T = [T; tmp];
end
writetable(T, 'D:\codeFrom0\statPeak\Table3_Statistics_Revised.xlsx');
fprintf('Table 3 exported: Table3_Statistics_Revised.xlsx\n');

%% === Console Summary ====================================================
fprintf('\n=== PEAK KAM STATISTICS (Holm-Bonferroni + Cohen''s d) ===\n');
for p = 1:nPeaks
    pk = peakNames{p};
    fprintf('\n--- %s ---\n', pk);
    fprintf('%-8s %8s %8s %8s %12s\n','Trial','p(raw)','p(adj)','d','95%% CI');
    for i = 1:nTrials
        fprintf('%-8s %8.4f %8.4f %8.3f [%6.3f, %6.3f]\n', ...
            trials{i}, results.(pk).p_raw(i), results.(pk).p_adj(i), ...
            results.(pk).cohen_d(i), results.(pk).CI_low(i), results.(pk).CI_high(i));
    end
end
% % 
