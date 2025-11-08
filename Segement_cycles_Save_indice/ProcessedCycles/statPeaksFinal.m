%  Description :
%  -----------------------------------------------------------------------
%  This script extracts the **first and second peaks** of the Knee
%  Adduction Moment (KAM) curves from predicted gait data.
%
%  It processes both **left** and **right** sides for a selected trial,
%  smooths the signals, identifies the peaks within specified gait-cycle
%  windows, plots the results, and saves peak values to a .mat file.
%
%  -----------------------------------------------------------------------
%  EXECUTION WORKFLOW
%  -----------------------------------------------------------------------
%  **First run (Pre-processing)**:
%      - Make sure you have run the grouping script that generates:
%        `trial##_Left.mat` and `trial##_Right.mat` files inside:
%        `ProcessedCyclesPrediction\Grouped\`
%      - Each file must contain:
%           → KAM_all_L (Left knee adduction moments)
%           → KAM_all_R (Right knee adduction moments)
%
%  **Second run (Peak extraction)**:
%      - Update the "trial" number in the Parameters section.
%      - Run this script to automatically:
%           • Load grouped data
%           • Smooth signals
%           • Extract first and second KAM peaks (early & late stance)
%           • Plot the results
%           • Save peaks in:
%             `ProcessedCyclesPrediction\peaks\Trial_HKA##_peaks.mat`
%
%  -----------------------------------------------------------------------
%  INPUTS
%  -----------------------------------------------------------------------
%  • trial##_Left.mat  → Contains KAM_all_L (subjects × time)
%  • trial##_Right.mat → Contains KAM_all_R (subjects × time)
%
%  -----------------------------------------------------------------------
%  OUTPUTS
%  -----------------------------------------------------------------------
%  • peak1_left, peak2_left, peak1_right, peak2_right
%       → Stored in `Trial_HKA##_peaks.mat`
%
%  -----------------------------------------------------------------------
%  USER PARAMETERS
%  -----------------------------------------------------------------------
%  → trial = Trial index to process
%  → x_line1:x_line4 = Define % gait cycle ranges for first and second peaks
%
%  -----------------------------------------------------------------------
%  AUTHOR & COPYRIGHT
%  -----------------------------------------------------------------------
%  Author      : Azza Tayari
%  Contact     : tayari.azza@eniso.u-sousse.tn

  clear; clc;close all ; 
% %
% 
% % === Load Trial 16 (reference) ===
refTrial = 32;
% refFile = sprintf('D:\\codeFrom0\\ProcessedCycles\\peaks\\Trial_HKA%d_peaks.mat', refTrial);
 refFile = sprintf('D:\\codeFrom0\\ProcessedCyclesPrediction\\peaks\\Trial_HKA%d_peaks.mat', refTrial);
data_ref = load(refFile);
nSubjects_ref = length(data_ref.peak1_left);

% === Parameters ===
peak_fields = {'peak1_left', 'peak2_left', 'peak1_right', 'peak2_right'};
peak_labels = {'Left Peak 1', 'Left Peak 2', 'Right Peak 1', 'Right Peak 2'};
trial_names = {};
trial_numbers = [];  % Store trial numbers for abbreviation
p_vals = [];
means = [];
stds = [];

% === Loop over all trials (excluding reference) ===
trial_idx = 1;
for t = 1:33
    if t == refTrial
        continue;
    end
%  file_path = sprintf('D:\\codeFrom0\\ProcessedCycles\\peaks\\Trial_HKA%d_peaks.mat', t);
   file_path = sprintf('D:\\codeFrom0\\ProcessedCyclesPrediction\\peaks\\Trial_HKA%d_peaks.mat', t);
    if exist(file_path, 'file')
        try
            data = load(file_path);
            % Check for presence of required fields
            if ~all(isfield(data, peak_fields))
                warning('Trial %d skipped: missing required peak fields.', t);
                continue;
            end
            
            % Determine minimum sample size between reference and current trial
            nSubjects_trial = length(data.peak1_left);
            minSamples = min(nSubjects_ref, nSubjects_trial);

            if minSamples < nSubjects_ref || minSamples < nSubjects_trial
                warning('Trial %d: truncating data to minimum sample size %d for fair comparison.', t, minSamples);
            end

            trial_names{trial_idx} = sprintf('Trial %d', t);
            trial_numbers(trial_idx) = t;  % save trial number

            for p = 1:length(peak_fields)
                ref_data_trunc = data_ref.(peak_fields{p})(1:minSamples);
                trial_data_trunc = data.(peak_fields{p})(1:minSamples);
                [~, p_vals(p, trial_idx)] = ttest(ref_data_trunc, trial_data_trunc);
                means(p, trial_idx+1) = mean(trial_data_trunc);
                stds(p, trial_idx+1)  = std(trial_data_trunc);
            end

            trial_idx = trial_idx + 1;

        catch ME
            warning('Error loading Trial %d: %s', t, ME.message);
        end
    else
        warning('File for Trial %d not found.', t);
    end
end

% === Add reference trial stats at column 1 ===
for p = 1:length(peak_fields)
    means(p,1) = mean(data_ref.(peak_fields{p}));
    stds(p,1)  = std(data_ref.(peak_fields{p}));
end

% === Grouping of Trials ===
conditionGroups = {
    'Speed Variation',      [2, 5, 8, 11, 14, 16];
    'Step Length Variation',[3, 6, 9, 10, 13];
    'Step Width Variation', [26, 27, 28, 29, 30];
};

% === Define fixed color for each trial abbreviation ===
trial_color_map = containers.Map(...
    {'Normal','VS','S','LN','F','M_F_r','VSSL','SSL','PSL','LSL','VLSL','SSW','MSW','WSW','VWSW','ISL'}, ...
    {
        [0.2 0.2 0.2],     % Normal
        [0 0.4470 0.7410], % VS
        [0.8500 0.3250 0.0980], % S
        [0.9290 0.6940 0.1250], % LN
        [0.4940 0.1840 0.5560], % F
        [0.4660 0.6740 0.1880], % M_F_r
        [0.3010 0.7450 0.9330], % VSSL
        [0.6350 0.0780 0.1840], % SSL
        [0.25 0.25 0.75],       % PSL
        [0.75 0.25 0.25],       % LSL
        [0.5 0.5 0],            % VLSL
        [0 0.5 0.5],            % SSW
        [0.4 0.4 0.4],          % MSW
        [0.3 0.7 0.3],          % WSW
        [0.7 0.3 0.7],          % VWSW
        [0.9 0.6 0.1]           % ISL
    });

% === Create grouped bar plots as subplots ===
figure('Name', 'Grouped KAM Comparisons', 'NumberTitle', 'off');
tiled = tiledlayout(3,1, 'TileSpacing','compact', 'Padding','compact');

for g = 1:size(conditionGroups,1)
    figTitle = conditionGroups{g,1};
    trialSet = conditionGroups{g,2};
    
    % Find trials in this group among loaded trials
    loaded_trial_nums = trial_numbers; % from loaded data (excluding ref)
    matchIdx = ismember(loaded_trial_nums, trialSet);
    matched_trials_idx = find(matchIdx);

    if isempty(matched_trials_idx)
        continue;
    end

    % Prepare data for group: include reference trial as first column
    means_group = [means(:,1), means(:, matched_trials_idx+1)];
    stds_group  = [stds(:,1),  stds(:, matched_trials_idx+1)];
    nbars = size(means_group, 2);
    ngroups = size(means_group, 1);

    % Trial abbreviations for legend
    abbrev_ref = getTrialAbbreviation(refTrial);
    abbrev_trials = cellfun(@getTrialAbbreviation, num2cell(loaded_trial_nums(matched_trials_idx)), 'UniformOutput', false);
    trial_labels = [{abbrev_ref}, abbrev_trials];

    % Generate color map
    cmap = lines(nbars);

    % === Subplot ===
    nexttile; % subplot (3,1,g)
   hb = bar(means_group, 'grouped'); hold on;

% Capture bar handles only once (first group)
if g == 1
    legendHandles = hb;
    legendLabels = trial_labels;
end
    % Apply colors
for k = 1:nbars
    abbrev = trial_labels{k};
    if trial_color_map.isKey(abbrev)
        hb(k).FaceColor = trial_color_map(abbrev);
    else
        hb(k).FaceColor = [0.5 0.5 0.5]; % fallback color if not defined
    end
end

    % Error bars
    groupwidth = min(0.8, nbars/(nbars + 1.5));
    for i = 1:nbars
        x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
        errorbar(x, means_group(:,i), stds_group(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1.2);
    end
%%
%     % Formatting
%     set(gca, 'XTick', 1:ngroups, 'XTickLabel', peak_labels);
% %     ylabel('Predection KAM  (Nm/Kg )');
%       ylabel('Normalized KAM (Nm/kg)');
%       title(['Group: ', figTitle]);
%      legend(trial_labels, 'Location', 'northeast');
%     grid on;

%%
set(gca, 'XTick', 1:ngroups, 'XTickLabel', peak_labels);
xlabel('Knee Adduction Moment Peaks (Left and Right)');
ylabel('Predicted Normalized KAM (Nm/kg)');
title(['Group: ', figTitle]);
legend(trial_labels, 'Location', 'northeast');
grid on;
set(gca, 'FontSize', 10, 'FontWeight', 'bold', 'Box', 'off');

%%

    % Significance stars
    sig_level = 0.05;
    for peakIdx = 1:ngroups
        for idx = 1:length(matched_trials_idx)
            trialIdx = matched_trials_idx(idx);  % index into p_vals columns
            pval = p_vals(peakIdx, trialIdx);
            x = peakIdx - groupwidth/2 + (2*(idx+1)-1) * groupwidth / (2*nbars);
            y = means_group(peakIdx, idx+1) + stds_group(peakIdx, idx+1) + 0.05 * max(means_group(:));
            if pval < sig_level
                text(x, y, '*', 'FontSize', 18, 'HorizontalAlignment', 'center', ...
                    'Color', 'r', 'FontWeight', 'bold');
            end
        end
    end

    hold off;
   
end

% === Save figure ===
save_filename = fullfile('D:\codeFrom0\ProcessedCyclesPrediction\plots', 'Grouped_Subplots_KAM.emf');
print(gcf, '-dmeta', save_filename);

%%

% === Display summary table of peak results ===
fprintf('\n=== Summary of Peak Knee Abduction Moments (Mean ± SD) ===\n');
header = sprintf('%-8s | %-10s | %-20s | %-20s | %-20s | %-20s', ...
    'Trial', 'Group', 'Left Peak 1', 'Left Peak 2', 'Right Peak 1', 'Right Peak 2');
disp(header);
disp(repmat('-', 1, length(header)));

% Include reference trial in output
all_trials = [refTrial, trial_numbers];
group_labels = strings(1, length(all_trials));
for i = 1:length(all_trials)
    t = all_trials(i);
    if ismember(t, conditionGroups{1,2})
        group = 'Speed';
    elseif ismember(t, conditionGroups{2,2})
        group = 'Step Length';
    elseif ismember(t, conditionGroups{3,2})
        group = 'Step Width';
    else
        group = 'Other';
    end
    group_labels(i) = group;

    % Display values
    m = means(:, i);
    s = stds(:, i);
    row = sprintf('%-8s | %-10s | %6.2f ± %-6.2f | %6.2f ± %-6.2f | %6.2f ± %-6.2f | %6.2f ± %-6.2f', ...
        getTrialAbbreviation(t), group, ...
        m(1), s(1), m(2), s(2), m(3), s(3), m(4), s(4));
    disp(row);
end



function abbreviation = getTrialAbbreviation(trialNumber)
    switch trialNumber
        case 1
            abbreviation = 'VSFr';
        case 2
            abbreviation = 'VS';
        case 3
            abbreviation = 'VSSL';
        case 4
            abbreviation = 'SFr';
        case 5
            abbreviation = 'S';
        case 6
            abbreviation = 'SSL';
        case 7
            abbreviation = 'MFr';
        case 8
            abbreviation = 'LN';
        case 9
            abbreviation = 'PSL';
        case 10
            abbreviation = 'LSL';
        case 11
            abbreviation = 'F';
        case 12
            abbreviation = 'VHFr';
        case 13
            abbreviation = 'VLSL';
        case 14
            abbreviation = 'M_F_r';
        case 15
            abbreviation = 'H_F_r';
        case 16
            abbreviation = 'VF';
        case 17
            abbreviation = '';  % No abbreviation assigned
        case 18
            abbreviation = '';  % No abbreviation assigned
        case 19
            abbreviation = '';  % No abbreviation assigned
        case 20
            abbreviation = '';  % No abbreviation assigned
        case 21
            abbreviation = '';  % No abbreviation assigned
        case 22
            abbreviation = '';  % No abbreviation assigned
        case 23
            abbreviation = '';  % No abbreviation assigned
        case 24
            abbreviation = '';  % No abbreviation assigned
        case 25
            abbreviation = '';  % No abbreviation assigned
        case 26
            abbreviation = 'VSSW';
        case 27
            abbreviation = 'SSW';
        case 28
            abbreviation = 'MSW';
        case 29
            abbreviation = 'WSW';
        case 30
            abbreviation = 'VWSW';
        case 31
            abbreviation = 'HFr';
        case 32
            abbreviation = 'Normal';
        case 33
            abbreviation = 'ISL';
        otherwise
            abbreviation = ''; % Default empty string for unknown trials
    end
end


