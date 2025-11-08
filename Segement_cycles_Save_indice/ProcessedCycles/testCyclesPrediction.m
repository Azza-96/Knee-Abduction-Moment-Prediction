%  Description:
%  -------------------------------------------------------------------------
%  This MATLAB script processes and organizes **predicted Knee Adduction
%  Moment (KAM)** and **Ground Reaction Force (GRF)** data obtained from a
%  gait prediction model. The script performs the following major steps:
%
%    1. Reads predicted left/right KAM CSV files and GRF CSV files.
%    2. Upsamples KAM data to improve temporal resolution.
%    3. Segments and normalizes individual gait cycles using the
%       `segment_and_normalize_gait_cycles` function.
%    4. Saves each normalized gait cycle (GRF + KAM) per trial and side.
%    5. Groups all cycles for each trial into combined `.mat` files
%       (Left / Right).
%    6. Generates a summary file containing the number of detected cycles.
%
%  -------------------------------------------------------------------------
%  Gait Conditions:
%  -------------------------------------------------------------------------
%  The dataset includes trials with different gait modification conditions:
%
%  • **Treadmill Speed Conditions:** [2, 5, 8, 11, 14, 16, 32]
%       Self-selected step length, frequency, and width at fixed speeds.
%
%  • **Variable Speed + Frequency:** [1, 4, 7, 12, 15, 31]
%       Varying treadmill speed and cadence while maintaining nominal step length.
%
%  • **Variable Step Length:** [3, 6, 9, 10, 13, 33]
%       Fixed treadmill speed; altered step length with constant frequency.
%
%  • **Variable Step Width:** [26, 27, 28, 29, 30]
%       Constant treadmill speed; altered step width conditions.
%
%  -------------------------------------------------------------------------
%  Outputs:
%  -------------------------------------------------------------------------
%  - Individual gait cycle `.mat` files:
%        → D:\codeFrom0\ProcessedCyclesPrediction\Left\cycles_[trial].mat
%        → D:\codeFrom0\ProcessedCyclesPrediction\Right\cycles_[trial].mat
%
%  - Grouped cycle files per trial:
%        → D:\codeFrom0\ProcessedCyclesPrediction\Grouped\trialXX_Left.mat
%        → D:\codeFrom0\ProcessedCyclesPrediction\Grouped\trialXX_Right.mat
%
%  - Summary of number of cycles:
%        → summary_cycle_counts.mat / summary_cycle_counts.csv
%
%  -------------------------------------------------------------------------
%  Notes:
%  -------------------------------------------------------------------------
%  • Make sure that the CSV input paths for GRF and predicted KAM are correct.
%  • The function `segment_and_normalize_gait_cycles.m` must be in the same directory
%    or in the MATLAB path.
%  • Normalization length = 100 samples per gait cycle.
%  • Minimum valid cycle length = 30 frames.
%
%  -------------------------------------------------------------------------
%  Dependencies:
%  -------------------------------------------------------------------------
%  - MATLAB R2021a or later
%  - Signal Processing Toolbox
%
%  -------------------------------------------------------------------------
%  © 2025 Azza Tayari. All rights reserved.

%% TREADMILL SPEED [2, 5, 8, 11, 16]; (done ) 
%  Tial Treadspeed	Step length 	Step frequency 	Step width 
% % 2	0.70 m.s-1	Self selected	Self selected	Self selected 
% % 5	0.90 m.s-1	Self selected	Self selected	Self selected
% % 8	1.10 m.s-1	Self selected	Self selected	Self selected
% % 11	1.60 m.s-1	Self selected	Self selected	Self selected
% % 14	1.80 m.s-1	Self selected	Self selected	Self selected
% % 16	2.20 m.s-1	Self selected	Self selected	Self selected
% % 32	1.40 m.s-1	Self selected	Self selected	Self selected
%% Variable speed + frequency [1,4,7,12,15,31]
% Tial 	Treadspeed	Step length   Step frequency 	Step width 
% % 1	0.70 m.s-1	1.00 s*	      0.56 f*	      Self selected 
% % 4	1.00 m.s-1	1.00 s*       0.72 f*	      Self selected
% % 7	1.10 m.s-1	1.00 s*	      0.88 f*	      Self selected
% % 12	1.60 m.s-1	1.00 s*	      1.28 f*	      Self selected
% % 15	1.80 m.s-1	1.00 s*       1.44 f* 	      Self selected
% % 31	1.40 m.s-1	1.00 s*	      1.12 f* 	      Self selected
%% Vriable step length [3,6,9,10,13,33]
% Tial 	Treadspeed	 Step length 	 Step frequency 	Step width 
% 3	     0.70 m.s-1 	0.56s*       1f*	          Self selected 
% 6	     0.90 m.s-1	    1s*	         1f*	          Self selected
% 9      1.10 m.s-1	    0.88 s*	     1f*	          Self selected
% 10	 1.60 m.s-1	    1.28 s*	     1f*	          Self selected
% 13	 1.80 m.s-1	    1.44 s*	     1f* 	          Self selected
% 33	 1.40 m.s-1	    1.12 s*	     1f* 	          Self selected

%% Vriable step Width [26,27,28,29,30]
% Tial 	Treadspeed	 Step length 	 Step frequency     Step width 
% 26	 1.25 m.s-1 	1s*      	    1f*	            Self selected 
% 27	 1.25 m.s-1	    1s*	            1f*	            Self selected
% 28	 1.25 m.s-1	    1s*	            1f*	            Self selected
% 29	 1.25 m.s-1	    1s*	            1f*	            Self selected
% 30	 1.25 m.s-1	    1s*	            1f*	            Self selected
close all ; clear all ; clc ; 
% Paths to CSV files

%% Initialization
close all; clear; clc;

%% File Paths – Predicted Data
Grf_L = 'D:\codeFrom0\ExtractedCSV\Validation\f1_axis3_validation.csv'; 
Grf_R = 'D:\codeFrom0\ExtractedCSV\Validation\f2_axis3_validation.csv';

% Predicted KAM CSVs (raw values)
L_KM_raw = readmatrix('D:\codeFrom0\all_trials_predicted_left_columns.csv');
R_KM_raw = readmatrix('D:\codeFrom0\all_trials_predicted_right_columns.csv');

%% Upsample KAM (factor 10)
L_KM = interp1(1:size(L_KM_raw,1), L_KM_raw, ...
               linspace(1, size(L_KM_raw,1), size(L_KM_raw,1)*10), 'linear');
R_KM = interp1(1:size(R_KM_raw,1), R_KM_raw, ...
               linspace(1, size(R_KM_raw,1), size(R_KM_raw,1)*10), 'linear');

%% Load GRF tables
Grf_Left_table  = readtable(Grf_L);
Grf_Right_table = readtable(Grf_R);
trialNames = Grf_Left_table.Properties.VariableNames;  % Trials

% Convert predicted KAM to tables with same headers
KAM_Left  = array2table(L_KM, 'VariableNames', trialNames);
KAM_Right = array2table(R_KM, 'VariableNames', trialNames);

% Trim GRF rows for alignment (from row 991 onward)
Grf_Left_table  = Grf_Left_table(991:end, :);
Grf_Right_table = Grf_Right_table(991:end, :);

%% Output Folders
output_folder_left  = 'D:\codeFrom0\ProcessedCyclesPrediction\Left';
output_folder_right = 'D:\codeFrom0\ProcessedCyclesPrediction\Right';
if ~exist(output_folder_left, 'dir'), mkdir(output_folder_left); end
if ~exist(output_folder_right, 'dir'), mkdir(output_folder_right); end

%% Parameters
min_cycle_length = 30;
norm_length = 100;

% Initialize output summary
numCycles_Left  = zeros(length(trialNames), 1);
numCycles_Right = zeros(length(trialNames), 1);

%% Loop through trials
for i = 1:length(trialNames)
    trial = trialNames{i};
    fprintf('🔄 Processing trial: %s\n', trial);

    %% ---- LEFT ----
    try
        grf_L = Grf_Left_table.(trial);
        kam_L = KAM_Left.(trial);

        [GRF_cycles_L, KAM_cycles_L, hs_indices_L] = ...
            segment_and_normalize_gait_cycles(grf_L, kam_L, ...
            min_cycle_length, norm_length);

        save(fullfile(output_folder_left, ['cycles_', trial, '.mat']), ...
            'GRF_cycles_L', 'KAM_cycles_L', 'hs_indices_L');

        numCycles_Left(i) = size(GRF_cycles_L, 2);
    catch ME
        warning('LEFT: Skipped trial %s due to error: %s', trial, ME.message);
    end

    %% ---- RIGHT ----
    try
        grf_R = Grf_Right_table.(trial);
        kam_R = KAM_Right.(trial);

        [GRF_cycles_R, KAM_cycles_R, hs_indices_R] = ...
            segment_and_normalize_gait_cycles(grf_R, kam_R, ...
            min_cycle_length, norm_length);

        save(fullfile(output_folder_right, ['cycles_', trial, '.mat']), ...
            'GRF_cycles_R', 'KAM_cycles_R', 'hs_indices_R');

        numCycles_Right(i) = size(GRF_cycles_R, 2);
    catch ME
        warning('RIGHT: Skipped trial %s due to error: %s', trial, ME.message);
    end
end

%% Save Summary
summary_table = table(trialNames', numCycles_Left, numCycles_Right, ...
    'VariableNames', {'Trial', 'NumCycles_Left', 'NumCycles_Right'});

save('D:\codeFrom0\ProcessedCyclesPrediction\summary_cycle_counts.mat', 'summary_table');
writetable(summary_table, 'D:\codeFrom0\ProcessedCyclesPrediction\summary_cycle_counts.csv');

disp('✅ All predicted cycles processed and saved.');


%% 2nd Run  ------------------- %%
%% ---- GROUP LEFT AND RIGHT PREDICTED CYCLES BY TRIAL ----

% Input & output paths
input_folder_left  = 'D:\codeFrom0\ProcessedCyclesPrediction\Left';
input_folder_right = 'D:\codeFrom0\ProcessedCyclesPrediction\Right';
grouped_output_folder = 'D:\codeFrom0\ProcessedCyclesPrediction\Grouped';

if ~exist(grouped_output_folder, 'dir')
    mkdir(grouped_output_folder);
end

% List all left-side files to extract unique trial numbers
left_files = dir(fullfile(input_folder_left, 'cycles_*.mat'));

% Extract trial numbers from filenames
trial_nums = [];
for k = 1:length(left_files)
    name = left_files(k).name;
    match = regexp(name, 'trial(\d+)', 'tokens');
    if ~isempty(match)
        trial_nums(end+1) = str2double(match{1}{1});
    end
end
unique_trials = unique(trial_nums);
norm_length = 100;  % Ensure consistent shape across trials

% Loop over each unique trial number
for t = unique_trials
    trial_str = ['trial', num2str(t)];

    %% ---- LEFT ----
    GRF_all_L = [];
    KAM_all_L = [];

    left_files_t = dir(fullfile(input_folder_left, ['cycles_*_', trial_str, '.mat']));
    for i = 1:length(left_files_t)
        data = load(fullfile(input_folder_left, left_files_t(i).name));
        if isfield(data, 'GRF_cycles_L') && size(data.GRF_cycles_L, 1) == norm_length
            GRF_all_L = [GRF_all_L, data.GRF_cycles_L];
            KAM_all_L = [KAM_all_L, data.KAM_cycles_L];
        end
    end

    if ~isempty(GRF_all_L)
        save(fullfile(grouped_output_folder, [trial_str, '_Left.mat']), ...
             'GRF_all_L', 'KAM_all_L');
    end

    %% ---- RIGHT ----
    GRF_all_R = [];
    KAM_all_R = [];

    right_files_t = dir(fullfile(input_folder_right, ['cycles_*_', trial_str, '.mat']));
    for i = 1:length(right_files_t)
        data = load(fullfile(input_folder_right, right_files_t(i).name));
        if isfield(data, 'GRF_cycles_R') && size(data.GRF_cycles_R, 1) == norm_length
            GRF_all_R = [GRF_all_R, data.GRF_cycles_R];
            KAM_all_R = [KAM_all_R, data.KAM_cycles_R];
        end
    end

    if ~isempty(GRF_all_R)
        save(fullfile(grouped_output_folder, [trial_str, '_Right.mat']), ...
             'GRF_all_R', 'KAM_all_R');
    end
end

disp('✅ Grouped predicted trials saved to Grouped folder.');
