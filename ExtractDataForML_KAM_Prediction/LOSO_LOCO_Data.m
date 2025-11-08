%% ================================================
%   STEP 0 — Folder Paths and Setup
% ================================================
folderPath = 'D:\codeFrom0\ExtractedTrialsMAT';
mainSaveFolder = 'D:\codeFrom0';

% Create main folder if needed
if ~exist(mainSaveFolder, 'dir'), mkdir(mainSaveFolder); end

% Constants
originalLength = 71980; % Original length at 1200 Hz
downsampleFactor = 10;  % 1200 Hz to 120 Hz (1200/120 = 10)
expectedLength = originalLength / downsampleFactor; % 7198 samples
numSubjects = 10;
numTrials = 33;

% Variables of interest
varNames = {'LLML_Acc', 'RLML_Acc', 'f1', 'f2', 'left_knee_moment', 'right_knee_moment'};
numAxes = 3;
numFields = length(varNames) * numAxes; % 18 fields

% Preallocate data and labels dynamically
allData = struct();
labels = struct();
for v = 1:length(varNames)
    for axis = 1:numAxes
        field = sprintf('%s_axis%d', varNames{v}, axis);
        allData.(field) = [];
        labels.(field) = {};
    end
end

% Index to track valid data columns
colIdx = 0;

%% ================================================
%   STEP 1 — Load and Downsample MAT Files
% ================================================
for subj = 1:numSubjects
    for trial = 1:numTrials
        fileName = sprintf('subj%d_trial%d.mat', subj, trial);
        fullPath = fullfile(folderPath, fileName);

        if exist(fullPath, 'file')
            S = load(fullPath);
            valid = true;

            % Validate required variables
            for v = 1:length(varNames)
                var = varNames{v};
                if ~isfield(S, var)
                    fprintf('Excluded: %s (missing variable: %s)\n', fileName, var);
                    valid = false;
                    break;
                elseif size(S.(var), 1) ~= originalLength
                    fprintf('Excluded: %s (invalid rows for %s: %d, expected %d)\n', ...
                        fileName, var, size(S.(var), 1), originalLength);
                    valid = false;
                    break;
                elseif size(S.(var), 2) < numAxes
                    fprintf('Excluded: %s (insufficient columns for %s: %d, expected >= %d)\n', ...
                        fileName, var, size(S.(var), 2), numAxes);
                    valid = false;
                    break;
                end
            end

            if valid
                colIdx = colIdx + 1;
                label = sprintf('subj%d_trial%d', subj, trial);

                for v = 1:length(varNames)
                    var = varNames{v};
                    data = S.(var);
                    % Downsample data (1200 Hz to 120 Hz)
                    downsampledData = downsample(data, downsampleFactor);
                    for axis = 1:numAxes
                        field = sprintf('%s_axis%d', var, axis);
                        allData.(field) = [allData.(field), downsampledData(:, axis)];
                        labels.(field) = [labels.(field), label];
                    end
                end
            end
        else
            fprintf('File not found: %s\n', fileName);
        end
    end
end

%% ================================================
% %   STEP 2 — Create Subject-Specific Test and Validation Splits
% % ================================================
% fprintf('\nCreating subject-specific datasets (subjX + LOSOsubjX in subjX folder)...\n');
% fields = fieldnames(allData);
% subjects = 1:numSubjects;
% 
% for s = subjects
%     % Create folder for the subject
%     subjFolder = fullfile(mainSaveFolder, sprintf('subj%d', s));
%     if ~exist(subjFolder, 'dir'), mkdir(subjFolder); end
% 
%     subjPattern = sprintf('subj%d_', s);
%     subjCols = contains(labels.(fields{1})(1:colIdx), subjPattern);
% 
%     for i = 1:length(fields)
%         field = fields{i};
%         data = allData.(field);
%         header = labels.(field);
% 
%         % Split
%         LOSOdata = data(:, ~subjCols);   % data excluding current subject
%         subjData = data(:, subjCols);    % data only for current subject
%         LOSOheader = header(~subjCols);
%         subjHeader = header(subjCols);
% 
%         % Define file names
%         subjFile = fullfile(subjFolder, sprintf('subj%d_%s.csv', s, field));
%         LOSOfile = fullfile(subjFolder, sprintf('LOSOsubj%d_%s.csv', s, field));
% 
%         % Save to CSV (subject data)
%         try
%             writecell([subjHeader; num2cell(subjData)], subjFile);
%             writecell([LOSOheader; num2cell(LOSOdata)], LOSOfile);
%             fprintf('Saved %s and %s\n', subjFile, LOSOfile);
%         catch e
%             fprintf('Error writing CSVs for subj%d - %s: %s\n', s, field, e.message);
%         end
%     end
% end
% 
% 
% 
% %%%
% 
% 
% outputFolder = 'path_to_save_LOCO_folders';     % Destination folder for LOCO results
% 
% % === Load all subject files ===
% subjectFiles = dir(fullfile(dataFolder, 'subj*.csv'));
% 
% % Collect all unique condition abbreviations dynamically
% allConditions = {};
% 
% for s = 1:length(subjectFiles)
%     T = readtable(fullfile(dataFolder, subjectFiles(s).name));
% 
%     % Ensure the column name matches your dataset
%     if ismember('LOCOtrialnumber', T.Properties.VariableNames)
%         conds = unique(T.LOCOtrialnumber);
%     elseif ismember('Condition', T.Properties.VariableNames)
%         conds = unique(T.Condition);
%     else
%         error('No "LOCOtrialnumber" or "Condition" column found in %s', subjectFiles(s).name);
%     end
%     
%     allConditions = [allConditions; conds];
% end
% 
% % Unique list of all condition abbreviations
% allConditions = unique(allConditions);
% 
% fprintf('Detected %d conditions:\n', numel(allConditions));
% disp(allConditions);
% 
% % === Generate LOCO folders and files ===
% for c = 1:length(allConditions)
%     cond = string(allConditions{c});
%     locoFolder = fullfile(outputFolder, ['LOCO_' cond]);
%     if ~exist(locoFolder, 'dir')
%         mkdir(locoFolder);
%     end
%     
%     allData = []; % Will store all data except the current condition
%     
%     for s = 1:length(subjectFiles)
%         subjFile = fullfile(dataFolder, subjectFiles(s).name);
%         T = readtable(subjFile);
%         
%         % Identify data for this condition
%         condRows = strcmp(T.LOCOtrialnumber, cond);
%         condData = T(condRows, :);
%         
%         % Save condition-specific subject file (if present)
%         if ~isempty(condData)
%             subjName = erase(subjectFiles(s).name, '.csv');
%             condFileName = sprintf('%s_%s.csv', subjName, cond);
%             writetable(condData, fullfile(locoFolder, condFileName));
%         end
%         
%         % Collect data for LOCO file (everything except this condition)
%         keepData = T(~condRows, :);
%         allData = [allData; keepData];
%     end
%     
%     % Save combined "leave-one-condition-out" dataset
%     locoFile = fullfile(locoFolder, ['LOCO_' cond '.csv']);
%     writetable(allData, locoFile);
%     
%     fprintf('✅ Created LOCO folder and files for condition: %s\n', cond);
% end
% 
% fprintf('\nAll LOCO datasets successfully generated.\n');

%% ================================================
%   STEP 3 — LOCO (Leave-One-Trial-Out)
% ================================================
fprintf('\nCreating LOCO datasets – leave-one-trial-out (trial 1,4,7,…)\n');

% ------------------------------------------------------------------
% 1. List of trials you want to leave out one-by-one
% ------------------------------------------------------------------
locoTrials = [1, 4, 7, 12, 15, 31, ...   % VariableSpeed
              3, 6, 9, 10, 13, 33, ...  % VariableStepLength
              26,27,28,29,30];         % VariableStepWidth

 fields = fieldnames(allData);   % 18 data fields
subjects = 1:numSubjects;

% ------------------------------------------------------------------
% 2. Loop over every trial to be excluded
% ------------------------------------------------------------------
for tIdx = 1:length(locoTrials)
    trialNum = locoTrials(tIdx);
    fprintf('Processing LOCO for trial %d ...\n', trialNum);

    % ----- create folder ------------------------------------------------
    locoFolder = fullfile(mainSaveFolder, sprintf('LOCO_trial%d', trialNum));
    if ~exist(locoFolder,'dir'), mkdir(locoFolder); end

    % ----- logical index of columns that belong to this trial ----------
    trialCols = false(1, colIdx);
    trialPattern = sprintf('_trial%d', trialNum);
    trialCols = contains(labels.(fields{1})(1:colIdx), trialPattern);

    % ----------------------------------------------------------------
    % 3. Save **trial-specific** file (only this trial, all subjects)
    % ----------------------------------------------------------------
    if any(trialCols)                                   % at least one subject has the trial
        for i = 1:length(fields)
            field = fields{i};
            data  = allData.(field);
            hdr   = labels.(field);

            trialData   = data(:, trialCols);
            trialHeader = hdr(trialCols);

            trialFile = fullfile(locoFolder, ...
                sprintf('trial%d_%s.csv', trialNum, field));

            try
                writecell([trialHeader; num2cell(trialData)], trialFile);
                fprintf('   Saved %s\n', trialFile);
            catch ME
                fprintf('   Error writing %s: %s\n', trialFile, ME.message);
            end
        end
    end

    % ----------------------------------------------------------------
    % 4. Save **LOCO** file = everything EXCEPT this trial
    % ----------------------------------------------------------------
    for i = 1:length(fields)
        field = fields{i};
        data  = allData.(field);
        hdr   = labels.(field);

        LOCOdata   = data(:, ~trialCols);
        LOCOheader = hdr(~trialCols);

        locoFile = fullfile(locoFolder, ...
            sprintf('LOCO_trial%d_%s.csv', trialNum, field));

        try
            writecell([LOCOheader; num2cell(LOCOdata)], locoFile);
            fprintf('   Saved %s\n', locoFile);
        catch ME
            fprintf('   Error writing %s: %s\n', locoFile, ME.message);
        end
    end
end

fprintf('\nAll LOCO_trialX folders created (trial-specific + LOCO files).\n');