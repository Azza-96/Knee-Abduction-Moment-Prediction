%  Description:
%  ---------------------------------------------------------------
%  This script extracts the **first and second knee adduction moment (KAM) peaks**
%  from both the left and right legs across all subjects for a given trial.
%  The peaks are identified from either ground truth data or predicted KAM
%  signals obtained during gait cycle analysis.
%
%  The KAM signals are smoothed, visualized, and processed to detect:
%     • First KAM peak (early stance, ~10–25% gait cycle)
%     • Second KAM peak (late stance, ~45–55% gait cycle)
%
%  Inputs:
%  ---------------------------------------------------------------
%  - Grouped KAM data for left and right legs (.mat files)
%       → KAM_all_L, GRF_all_L  (Left leg)
%       → KAM_all_R, GRF_all_R  (Right leg)
%
%  Output:
%  ---------------------------------------------------------------
%  - A .mat file containing the extracted peak magnitudes for each subject:
%       → peak1_left, peak2_left, peak1_right, peak2_right
%
%  - Visual plots of smoothed KAM curves with detected peaks highlighted.
%
%  File naming convention:
%  ---------------------------------------------------------------
%     Input files:
%       D:\codeFrom0\ProcessedCyclesPrediction\Grouped\trialXX_Left.mat
%       D:\codeFrom0\ProcessedCyclesPrediction\Grouped\trialXX_Right.mat
%
%     Output file:
%       D:\codeFrom0\ProcessedCyclesPrediction\peaks\Trial_HKAXX_peaks.mat
%
%  Notes:
%  ---------------------------------------------------------------
%  - The script can be adapted for both ground truth and predicted data
%    by changing the input/output folder paths.
%  - The peak detection is based on prominence and fallback to max value
%    within defined gait cycle windows if no peak is found.
%
%  Dependencies:
%  ---------------------------------------------------------------
%  - MATLAB R2021a or newer
%  - Signal Processing Toolbox (for smoothdata, findpeaks)
%
%  © 2025 Azza Tayari. All rights reserved.
clear; close all; clc;
%% ======= Parameters =======
trial =27;  % Change trial number here
%% Original Dat 
% outputPath = 'D:\codeFrom0\ProcessedCycles\peaks';
 outputPath = 'D:\codeFrom0\ProcessedCyclesPrediction\peaks';
if ~exist(outputPath, 'dir')
    mkdir(outputPath);
end

% Load grouped trial data
%% Original 
% leftFile = fullfile('D:\codeFrom0\ProcessedCycles\Grouped', sprintf('trial%d_Left.mat', trial));
% rightFile = fullfile('D:\codeFrom0\ProcessedCycles\Grouped', sprintf('trial%d_Right.mat', trial));
%% Prediction
leftFile = fullfile('D:\codeFrom0\ProcessedCyclesPrediction\Grouped', sprintf('trial%d_Left.mat', trial));
rightFile = fullfile('D:\codeFrom0\ProcessedCyclesPrediction\Grouped', sprintf('trial%d_Right.mat', trial));

load(leftFile);   % Loads: GRF_all_L, KAM_all_L
load(rightFile);  % Loads: GRF_all_R, KAM_all_R

% Set gait cycle time vector
nSubjects = min(size(KAM_all_L, 2), size(KAM_all_R, 2));  % Ensure alignment
y = linspace(0, 100, size(KAM_all_L, 1));  % Gait cycle 0–100%

% Detection windows (% gait cycle)
x_line1 = 10;   % First peak start
x_line2 = 25;  % First peak end
x_line3 = 45;  % Second peak start
x_line4 = 55;  % Second peak end

% Initialize peak storage
peak1_left = zeros(nSubjects, 1);
peak2_left = zeros(nSubjects, 1);
peak1_right = zeros(nSubjects, 1);
peak2_right = zeros(nSubjects, 1);

figure;

for i = 1:nSubjects
    signal_L = KAM_all_L(:, i);
    signal_R = KAM_all_R(:, i);

    % Smooth signals
    smoothed_L = smoothdata(abs(signal_L), 'movmean', 5);
    smoothed_R = smoothdata(abs(signal_R), 'movmean', 5);

    % Plot
    subplot(2,1,1); hold on;
    plot(y, smoothed_L, 'b', 'LineWidth', 1.5);
    title('Left Knee Abduction Moment');
    ylabel('Moment (Nm)');
    grid on;

    subplot(2,1,2); hold on;
    plot(y, smoothed_R, 'r', 'LineWidth', 1.5);
    title('Right Knee Abduction Moment');
    xlabel('Gait Cycle (%)');
    ylabel('Moment (Nm)');
    grid on;

    % Extract peaks
    [pk1_L, loc1_L, pk2_L, loc2_L] = extract_peaks(smoothed_L, y, x_line1, x_line2, x_line3, x_line4);
    [pk1_R, loc1_R, pk2_R, loc2_R] = extract_peaks(smoothed_R, y, x_line1, x_line2, x_line3, x_line4);

    % Plot peaks
    subplot(2,1,1);
    plot(loc1_L, pk1_L, 'ko', 'MarkerSize', 6, 'LineWidth', 1.5);
    plot(loc2_L, pk2_L, 'mo', 'MarkerSize', 6, 'LineWidth', 1.5);

    subplot(2,1,2);
    plot(loc1_R, pk1_R, 'ko', 'MarkerSize', 6, 'LineWidth', 1.5);
    plot(loc2_R, pk2_R, 'mo', 'MarkerSize', 6, 'LineWidth', 1.5);

    % Store peaks
    peak1_left(i) = pk1_L;
    peak2_left(i) = pk2_L;
    peak1_right(i) = pk1_R;
    peak2_right(i) = pk2_R;
end

% Save peak data
saveFile = fullfile(outputPath, sprintf('Trial_HKA%d_peaks.mat', trial));
save(saveFile, 'peak1_left', 'peak2_left', 'peak1_right', 'peak2_right');
fprintf('✅ Saved peaks for Trial %d to %s\n', trial, saveFile);

%% ======= Helper Function =======
function [pk_val1, pk_loc1, pk_val2, pk_loc2] = extract_peaks(signal, y, x1, x2, x3, x4)
    % First peak: early stance
    idx1 = y >= x1 & y <= x2;
    [pks1, locs1] = findpeaks(signal(idx1), y(idx1), 'MinPeakProminence', 0.1);
    if isempty(pks1)
        [pks1, idx] = max(signal(idx1));
        locs_temp = y(idx1);
        locs1 = locs_temp(idx);
    end
    pk_val1 = pks1;
    pk_loc1 = locs1;

    % Second peak: late stance
    idx2 = y >= x3 & y <= x4;
    [pks2, locs2] = findpeaks(signal(idx2), y(idx2), 'MinPeakProminence', 0.1);
    if isempty(pks2)
        [pks2, idx] = max(signal(idx2));
        locs_temp = y(idx2);
        locs2 = locs_temp(idx);
    end
    pk_val2 = pks2;
    pk_loc2 = locs2;
end
