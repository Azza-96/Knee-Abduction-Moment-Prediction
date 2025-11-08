function [GRF_cycles, KneeMom_cycles, heel_strike_indices] = segment_and_normalize_gait_cycles(grf_signal, knee_moment_signal, min_cycle_length, norm_length)

    if nargin < 3
        min_cycle_length = 30;
    end
    if nargin < 4
        norm_length = 100;
    end

    %% Step 1: Detect heel strikes
    threshold = -0.2 * std(grf_signal);
    binary_contact = grf_signal < threshold;
    heel_strike_indices_raw = find(diff(binary_contact) == 1);  % rising edge

    %% Step 2: Extract and normalize cycles
    num_cycles = length(heel_strike_indices_raw) - 1;
    GRF_raw = {};
    KAM_raw = {};
    valid_indices = [];

    for i = 1:num_cycles
        idx1 = heel_strike_indices_raw(i);
        idx2 = heel_strike_indices_raw(i+1);

        if (idx2 - idx1) >= min_cycle_length
            % Segment and interpolate
            grf_segment = grf_signal(idx1:idx2);
            kam_segment = knee_moment_signal(idx1:idx2);

            t_orig = linspace(0, 1, length(grf_segment));
            t_norm = linspace(0, 1, norm_length);

            grf_interp = interp1(t_orig, grf_segment, t_norm, 'spline');
            kam_interp = interp1(t_orig, kam_segment, t_norm, 'spline');

            GRF_raw{end+1} = grf_interp(:);
            KAM_raw{end+1} = kam_interp(:);
            valid_indices(end+1) = idx1;
        end
    end

    %% Step 3: Compute pairwise correlations between KAM cycles
    num_valid = length(KAM_raw);
    if num_valid < 8
        warning('Only %d valid cycles found. Returning all.', num_valid);
        N = num_valid;
    else
        N = 8;
    end

    if num_valid == 0
        GRF_cycles = [];
        KneeMom_cycles = [];
        heel_strike_indices = [];
        return;
    end

    % Create a matrix where each column is a KAM cycle
    KAM_mat = cell2mat(KAM_raw');
    % Compute pairwise correlation matrix
    corr_mat = corr(KAM_mat);

    % Compute mean correlation per cycle
    mean_corr = mean(corr_mat - eye(num_valid), 2);  % exclude self-correlation
    [~, best_idxs] = maxk(mean_corr, N);

    %% Step 4: Keep only top N most correlated cycles
    GRF_cycles = cell2mat(GRF_raw(best_idxs)');
    GRF_cycles = reshape(GRF_cycles, norm_length, N);

    KneeMom_cycles = cell2mat(KAM_raw(best_idxs)');
    KneeMom_cycles = reshape(KneeMom_cycles, norm_length, N);

    heel_strike_indices = valid_indices(best_idxs)';
end
