% MATLAB script to extract and save biomechanical data for KAM prediction
% Purpose: Processes .mat files containing motion capture data for 10 subjects,
% extracts knee, hip, ankle moments, and foot acceleration data, and saves them
% as individual .mat files for machine learning (KAM prediction).
%
% Instructions to Run:
% 1. Ensure MATLAB is installed with required toolboxes (e.g., Signal Processing).
% 2. Place raw .mat files in 'D:\codeFrom0\ExtractDataForML_KAM_Predection\V3D exported data'
% 3. Verify the output directory 'D:\codeFrom0\ExtractedTrialsMAT' is accessible
%    or update 'outputDirectory' if needed.
% 4. Ensure the 'Extract_Filter_knee_Hip_Ankle_Data' function is in the MATLAB path.
% 5. Run the script in MATLAB. It will process trials for subjects P1 to P10,
%    save results as 'subj<subject>_trial<trial>.mat', and report success/failure.
%
% Notes:
% - The script assumes 33 trials per subject (330 total files).
% - Errors are logged for failed trials, and a summary is displayed at the end.
% - Adjust 'parentDirectory' or 'outputDirectory' if file paths differ.
%
% Copyright Azza Tayari 2025 

clear all; close all; clc;

% Input and output directories
parentDirectory = 'D:\codeFrom0\ExtractDataForML_KAM_Predection\V3D exported data';
outputDirectory = 'D:\codeFrom0\ExtractedTrialsMAT';

% Create output directory if it doesn't exist
if ~exist(outputDirectory, 'dir')
    mkdir(outputDirectory);
end

% Counters for saved and failed files
savedCount = 0;
failedTrials = {};

% Loop through subjects
for p = 1:10
    % Input directory for this subject
    inputDir = fullfile(parentDirectory, ['P', num2str(p), 'exportedfiles']);

    % Get all .mat trial files matching pattern
    contents = dir(inputDir);
    filePattern = '^p\d+export_T\d+\.mat$';
    fileIndices = ~[contents.isdir] & cellfun(@(x) ~isempty(x), regexp({contents.name}, filePattern));
    trialFiles = contents(fileIndices);

    fprintf('📂 P%dexportedfiles - Found %d trials\n', p, numel(trialFiles));

    % Loop through trials
    for i = 1:length(trialFiles)
        filename = trialFiles(i).name;
        filepath = fullfile(inputDir, filename);

        try
            % Extract data
            [left_knee_moment, f1, right_knee_moment, f2, ...
             left_hip_moment, right_hip_moment, ...
             left_ankle_moment, right_ankle_moment, ...
             LLML_Acc, RLML_Acc] = Extract_Filter_knee_Hip_Ankle_Data(filepath);

            % Output filename
            matFilename = sprintf('subj%d_trial%d.mat', p, i);
            fullOutputPath = fullfile(outputDirectory, matFilename);

            % Save all variables into a .mat file
            save(fullOutputPath, ...
                 'left_knee_moment', 'f1', 'right_knee_moment', 'f2', ...
                 'left_hip_moment', 'right_hip_moment', ...
                 'left_ankle_moment', 'right_ankle_moment', ...
                 'LLML_Acc', 'RLML_Acc');

            fprintf('✅ Saved: subj%d_trial%d.mat\n', p, i);
            savedCount = savedCount + 1;

        catch ME
            fprintf('❌ Error processing subj%d_trial%d (%s)\nReason: %s\n', ...
                    p, i, filename, ME.message);
            failedTrials{end+1} = sprintf('subj%d_trial%d', p, i); %#ok<*SAGROW>
        end
    end
end

% Final summary
fprintf('\n🔔 Done.\nExpected files: %d\nSuccessfully saved: %d\nFailed trials: %d\n', ...
        10 * 33, savedCount, numel(failedTrials));

if ~isempty(failedTrials)
    fprintf('\n❌ Failed Trials:\n');
    disp(failedTrials');
end
