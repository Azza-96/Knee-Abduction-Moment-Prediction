
%  Description:
%  -------------------------------------------------------------------------
%  This MATLAB script processes **original experimental Ground Reaction Force (GRF)**
%  and **Knee Adduction Moment (KAM)** data for all treadmill walking trials.
%  It segments each continuous trial signal into individual gait cycles and
%  normalizes each cycle to 100 points (0–100% gait cycle).
%
%  The script performs the following operations:
%
%    1. Loads original GRF and KAM CSV data for left and right limbs.
%    2. Defines trial information and experimental conditions:
%         • Treadmill speed trials (self-selected step parameters)
%         • Variable speed and frequency trials
%         • Variable step length trials
%         • Variable step width trials
%    3. Segments GRF and KAM signals into gait cycles using
%       the `segment_and_normalize_gait_cycles` function.
%    4. Saves each segmented trial (per side) as a `.mat` file.
%    5. Generates a summary of detected gait cycles per trial.
%    6. (Optional) Groups all cycles by trial and saves combined `.mat` files.
%
%  -------------------------------------------------------------------------
%  Input Data:
%  -------------------------------------------------------------------------
%  Path: D:\codeFrom0\ExtractedCSV\Validation\
%
%      • f1_axis3_validation.csv   → GRF (Left foot)
%      • f2_axis3_validation.csv   → GRF (Right foot)
%      • left_knee_moment_axis2_validation.csv   → KAM (Left)
%      • right_knee_moment_axis2_validation.csv  → KAM (Right)
%
%  -------------------------------------------------------------------------
%  Output Folders:
%  -------------------------------------------------------------------------
%  • D:\codeFrom0\ProcessedCycles\Left\      → Individual Left cycles
%  • D:\codeFrom0\ProcessedCycles\Right\     → Individual Right cycles
%  • D:\codeFrom0\ProcessedCycles\Grouped\   → (Optional) Grouped by trial
%  • D:\codeFrom0\ProcessedCycles\summary_cycle_counts.csv
%
%  -------------------------------------------------------------------------
%  Experimental Conditions:
%  -------------------------------------------------------------------------
%  **Treadmill Speed Trials**  → [2, 5, 8, 11, 14, 16, 32]
%     - Step length, frequency, and width: Self-selected
%
%  **Variable Speed + Frequency**  → [1, 4, 7, 12, 15, 31]
%     - Step length fixed (1.00 s*)
%     - Step frequency variable (0.56–1.44 f*)
%
%  **Variable Step Length**  → [3, 6, 9, 10, 13, 33]
%     - Step length varied (0.56–1.44 s*)
%     - Step frequency fixed (1 f*)
%
%  **Variable Step Width**  → [26, 27, 28, 29, 30]
%     - Constant treadmill speed (1.25 m·s⁻¹)
%     - Step length and frequency fixed (1 s*, 1 f*)
%
%  -------------------------------------------------------------------------
%  Parameters:
%  -------------------------------------------------------------------------
%     • min_cycle_length = 30   (minimum valid samples per cycle)
%     • norm_length      = 100  (normalized points per gait cycle)
%
%  -------------------------------------------------------------------------
%  Dependencies:
%  -------------------------------------------------------------------------
%     - MATLAB R2021a or later
%     - Custom function: segment_and_normalize_gait_cycles.m
%     - Signal Processing Toolbox
%
%  -------------------------------------------------------------------------
%  Notes:
%  -------------------------------------------------------------------------
%  • Ensure that CSV column names correspond to valid trial identifiers.
%  • Adjust paths according to your local data directory.
%  • Optional plotting sections are included for verification.
%  • Grouping section (commented) can be enabled for multi-subject merging.
%
%  -------------------------------------------------------------------------
%  © 2025 Azza Tayari. All rights reserved.

close all ; clear all ; clc ; 
% Paths to CSV files

% THIS TO LOAD ORIGINAL DATA 
Grf_L = 'D:\codeFrom0\ExtractedCSV\Validation\f1_axis3_validation.csv'; 
Grf_R = 'D:\codeFrom0\ExtractedCSV\Validation\f2_axis3_validation.csv';

L_KM  = 'D:\codeFrom0\ExtractedCSV\Validation\left_knee_moment_axis2_validation.csv';
R_KM  = 'D:\codeFrom0\ExtractedCSV\Validation\right_knee_moment_axis2_validation.csv';
% % Read tables
Grf_Left   = readtable(Grf_L);
Grf_Right  = readtable(Grf_R);
KAM_Left   = readtable(L_KM);
KAM_Right  = readtable(R_KM);
%%
% figure 
%  plot(Grf_Right(:,1))
%  hold on 
%  plot(R_KM(:,1))
%% Oringinal 

% Trial names (column headers)
 trialNames = Grf_Left.Properties.VariableNames;

 %%

% Parameters
min_cycle_length = 30;
norm_length = 100;
% Original 
Output folders
output_folder_left  = 'D:\codeFrom0\ProcessedCycles\Left';
output_folder_right = 'D:\codeFrom0\ProcessedCycles\Right';


if ~exist(output_folder_left, 'dir'), mkdir(output_folder_left); end
if ~exist(output_folder_right, 'dir'), mkdir(output_folder_right); end

% Initialize arrays to save cycle counts
numCycles_Left  = zeros(length(trialNames), 1);
numCycles_Right = zeros(length(trialNames), 1);

% Loop through each trial
for i = 1:length(trialNames)
    trial = trialNames{i};

    %% ---- LEFT ----
    try
        grf_L = Grf_Left.(trial);
        kam_L = KAM_Left.(trial);

        [GRF_cycles_L, KAM_cycles_L, hs_indices_L] = segment_and_normalize_gait_cycles( ...
            grf_L, kam_L, min_cycle_length, norm_length);

        % Save
        save(fullfile(output_folder_left, ['cycles_', trial, '.mat']), ...
            'GRF_cycles_L', 'KAM_cycles_L', 'hs_indices_L');

        % Store number of cycles
        numCycles_Left(i) = size(GRF_cycles_L, 2);

% %         Optional plot
%         figure('Name', ['LEFT - ', trial], 'NumberTitle', 'off');
%         subplot(2,1,1); plot(GRF_cycles_L(:, 1:min(3,end))); title(['GRF Left - ', trial]);
%         subplot(2,1,2); plot(KAM_cycles_L(:, 1:min(3,end))); title(['KAM Left - ', trial]);

    catch ME
        warning('LEFT: Skipped trial %s due to error: %s', trial, ME.message);
    end

    %% ---- RIGHT ----
    try
        grf_R = Grf_Right.(trial);
        kam_R = KAM_Right.(trial);

        [GRF_cycles_R, KAM_cycles_R, hs_indices_R] = segment_and_normalize_gait_cycles( ...
            grf_R, kam_R, min_cycle_length, norm_length);

        % Save
        save(fullfile(output_folder_right, ['cycles_', trial, '.mat']), ...
            'GRF_cycles_R', 'KAM_cycles_R', 'hs_indices_R');

        % Store number of cycles
        numCycles_Right(i) = size(GRF_cycles_R, 2);

%         % Optional plot
%         figure('Name', ['RIGHT - ', trial], 'NumberTitle', 'off');
%         subplot(2,1,1); plot(GRF_cycles_R(:, 1:min(3,end))); title(['GRF Right - ', trial]);
%         subplot(2,1,2); plot(KAM_cycles_R(:, 1:min(3,end))); title(['KAM Right - ', trial]);

    catch ME
        warning('RIGHT: Skipped trial %s due to error: %s', trial, ME.message);
    end
end

%% ---- Save Summary Table ----
summary_table = table(trialNames', numCycles_Left, numCycles_Right, ...
    'VariableNames', {'Trial', 'NumCycles_Left', 'NumCycles_Right'});

save('D:\codeFrom0\ProcessedCycles\summary_cycle_counts.mat', 'summary_table');
writetable(summary_table, 'D:\codeFrom0\ProcessedCycles\summary_cycle_counts.csv');

disp('✅ All cycles processed and saved.');

% %% 2nd Run  ------------------- %%
% %% ---- GROUP LEFT AND RIGHT BY TRIAL ----
% %% Original Data 
% % % Paths
% % input_folder_left  = 'D:\codeFrom0\ProcessedCycles\Left';
% % input_folder_right = 'D:\codeFrom0\ProcessedCycles\Right';
% % grouped_output_folder = 'D:\codeFrom0\ProcessedCycles\Grouped';
% %%
% if ~exist(grouped_output_folder, 'dir'), mkdir(grouped_output_folder); end
% 
% % List all left-side mat files
% left_files = dir(fullfile(input_folder_left, 'cycles_subj*_trial*.mat'));
% 
% % Extract trial numbers from filenames
% trial_nums = [];
% for k = 1:length(left_files)
%     name = left_files(k).name;
%     match = regexp(name, 'trial(\d+)', 'tokens');
%     if ~isempty(match)
%         trial_nums(end+1) = str2double(match{1}{1});
%     end
% end
% unique_trials = unique(trial_nums);
% norm_length = 100;  % Make sure it matches your processing
% 
% for t = unique_trials
%     trial_str = ['trial', num2str(t)];
% 
%     %% ---- LEFT ----
%     GRF_all_L = [];
%     KAM_all_L = [];
% 
%     left_files_t = dir(fullfile(input_folder_left, ['cycles_subj*_', trial_str, '.mat']));
%     for i = 1:length(left_files_t)
%         data = load(fullfile(input_folder_left, left_files_t(i).name));
%         if isfield(data, 'GRF_cycles_L') && size(data.GRF_cycles_L, 1) == norm_length
%             GRF_all_L = [GRF_all_L, data.GRF_cycles_L];
%             KAM_all_L = [KAM_all_L, data.KAM_cycles_L];
%         end
%     end
% 
%     if ~isempty(GRF_all_L)
%         save(fullfile(grouped_output_folder, [trial_str, '_Left.mat']), ...
%             'GRF_all_L', 'KAM_all_L');
%     end
% 
%     %% ---- RIGHT ----
%     GRF_all_R = [];
%     KAM_all_R = [];
% 
%     right_files_t = dir(fullfile(input_folder_right, ['cycles_subj*_', trial_str, '.mat']));
%     for i = 1:length(right_files_t)
%         data = load(fullfile(input_folder_right, right_files_t(i).name));
%         if isfield(data, 'GRF_cycles_R') && size(data.GRF_cycles_R, 1) == norm_length
%             GRF_all_R = [GRF_all_R, data.GRF_cycles_R];
%             KAM_all_R = [KAM_all_R, data.KAM_cycles_R];
%         end
%     end
% 
%     if ~isempty(GRF_all_R)
%         save(fullfile(grouped_output_folder, [trial_str, '_Right.mat']), ...
%             'GRF_all_R', 'KAM_all_R');
%     end
% end
% 
% disp('✅ Grouped trials saved to Grouped folder.');
% 
