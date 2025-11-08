% MATLAB script to convert biomechanical .mat files to CSV for KAM prediction
% % % ExtractedCSV/
% % % ├── Test/
% % % │   ├── LLML_Acc_axis1_test.csv
% % % │   ├── ...
% % % ├── Validation/
% % % │   ├── LLML_Acc_axis1_validation.csv
% % % │   ├── ...
% Instructions to Run:
% 1. Ensure MATLAB is installed with required toolboxes (e.g., Signal Processing).
% 2. Input: Place .mat files (subj<1-10>_trial<1-33>.mat) in 'D:\codeFrom0\ExtractedTrialsMAT'.
% 3. Output: CSV files will be saved in 'D:\codeFrom0\ExtractedCSV\Test' and 'D:\codeFrom0\ExtractedCSV\Validation'.
% 4. Run the script in MATLAB to process all trials, generating test (first 50,000 rows) and validation (remaining rows) CSVs for each variable (e.g., LLML_Acc_axis1, left_knee_moment_axis1).

% Copyright Azza Tayari 2025

% Set folder paths 
folderPath = 'D:\codeFrom0\ExtractedTrialsMAT';
mainSaveFolder = 'D:\codeFrom0\ExtractedCSV';
testFolder = fullfile(mainSaveFolder, 'Test');
validationFolder = fullfile(mainSaveFolder, 'Validation');

% Constants
expectedLength = 71980;
numSubjects = 10;
numTrials = 33;

% Create folders if needed
if ~exist(mainSaveFolder, 'dir'), mkdir(mainSaveFolder); end
if ~exist(testFolder, 'dir'), mkdir(testFolder); end
if ~exist(validationFolder, 'dir'), mkdir(validationFolder); end

% Variables of interest
varNames = {'LLML_Acc', 'RLML_Acc', 'f1', 'f2', 'left_knee_moment', 'right_knee_moment'};

% Precompute maximum number of columns = numSubjects * numTrials
maxCols = numSubjects * numTrials;

% Preallocate data and labels
allData = struct();
labels = struct();
for v = 1:length(varNames)
    for axis = 1:3
        field = sprintf('%s_axis%d', varNames{v}, axis);
        allData.(field) = nan(expectedLength, maxCols);
        labels.(field) = cell(1, maxCols);
    end
end

% Index to track valid data columns
colIdx = 0;

% Loop over all subjects and trials
for subj = 1:numSubjects
    for trial = 1:numTrials
        fileName = sprintf('subj%d_trial%d.mat', subj, trial);
        fullPath = fullfile(folderPath, fileName);

        if exist(fullPath, 'file')
            S = load(fullPath);
            valid = true;

            % Validate required variables exist and are correct shape
            for v = 1:length(varNames)
                var = varNames{v};
                if ~isfield(S, var) || size(S.(var), 1) ~= expectedLength || size(S.(var), 2) < 3
                    valid = false;
                    fprintf('Excluded: %s (invalid: %s)\n', fileName, var);
                    break;
                end
            end

            if valid
                colIdx = colIdx + 1;
                label = sprintf('subj%d_trial%d', subj, trial);

                for v = 1:length(varNames)
                    var = varNames{v};
                    data = S.(var);
                    for axis = 1:3
                        field = sprintf('%s_axis%d', var, axis);
                        allData.(field)(:, colIdx) = data(:, axis);
                        labels.(field){colIdx} = label;
                    end
                end
            end
        else
            fprintf('File not found: %s\n', fileName);
        end
    end
end

% Trim and save to CSV
fields = fieldnames(allData);
for i = 1:length(fields)
    field = fields{i};
    data = allData.(field)(:, 1:colIdx);
    header = labels.(field)(1:colIdx);

    % Save test data (first 50,000 rows)
    testData = [header; num2cell(data(1:50000, :))];
    writecell(testData, fullfile(testFolder, [field '_test.csv']));

    % Save validation data (remaining rows)
    validationData = [header; num2cell(data(50001:end, :))];
    writecell(validationData, fullfile(validationFolder, [field '_validation.csv']));

    fprintf('Saved:\n→ %s\n→ %s\n', ...
        fullfile(testFolder, [field '_test.csv']), ...
        fullfile(validationFolder, [field '_validation.csv']));
end

