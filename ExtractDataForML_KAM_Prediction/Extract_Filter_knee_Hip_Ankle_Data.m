function [left_knee_moment, f1, right_knee_moment, f2, ...
          left_hip_moment, right_hip_moment, ...
          left_ankle_moment, right_ankle_moment, ...
          LLML_Acc, RLML_Acc] = Extract_Filter_knee_Hip_Ankle_Data(A)
% This function extracts left and right knee/hip/ankle moments and GRFs,
% filters them using SGOLAY filter, and synchronizes acceleration data.
    
    % Load data
    load(char(A));

    % SGOLAY filter setup
    order = 5;
    framelen = 21;
    b = sgolay(order, framelen);

    %% Knee Moments
    l_knee_moment_x = conv(l_kne_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    l_knee_moment_y = conv(l_kne_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    l_knee_moment_z = conv(l_kne_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    r_knee_moment_x = conv(r_kne_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    r_knee_moment_y = conv(r_kne_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    r_knee_moment_z = conv(r_kne_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    %% Hip Moments
    l_hip_moment_x = conv(l_hip_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    l_hip_moment_y = conv(l_hip_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    l_hip_moment_z = conv(l_hip_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    r_hip_moment_x = conv(r_hip_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    r_hip_moment_y = conv(r_hip_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    r_hip_moment_z = conv(r_hip_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    %% Ankle Moments
    l_ankle_moment_x = conv(l_ank_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    l_ankle_moment_y = conv(l_ank_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    l_ankle_moment_z = conv(l_ank_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    r_ankle_moment_x = conv(r_ank_moment{1,1}(:,1), b((framelen+1)/2,:), 'valid');
    r_ankle_moment_y = conv(r_ank_moment{1,1}(:,2), b((framelen+1)/2,:), 'valid');
    r_ankle_moment_z = conv(r_ank_moment{1,1}(:,3), b((framelen+1)/2,:), 'valid');

    %% GRFs
    fx_1 = conv(f1xprocessed{1,1}, b((framelen+1)/2,:), 'valid');
    fy_1 = conv(f1yprocessed{1,1}, b((framelen+1)/2,:), 'valid');
    fz_1 = conv(f1zprocessed{1,1}, b((framelen+1)/2,:), 'valid');

    fx_2 = conv(f2xprocessed{1,1}, b((framelen+1)/2,:), 'valid');
    fy_2 = conv(f2yprocessed{1,1}, b((framelen+1)/2,:), 'valid');
    fz_2 = conv(f2zprocessed{1,1}, b((framelen+1)/2,:), 'valid');

    f1 = [fx_1, fy_1, fz_1];
    f2 = [fx_2, fy_2, fz_2];

    %% Group Moments
    l_k_m = [l_knee_moment_x, l_knee_moment_y, l_knee_moment_z];
    r_k_m = [r_knee_moment_x, r_knee_moment_y, r_knee_moment_z];
    l_h_m = [l_hip_moment_x, l_hip_moment_y, l_hip_moment_z];
    r_h_m = [r_hip_moment_x, r_hip_moment_y, r_hip_moment_z];
    l_a_m = [l_ankle_moment_x, l_ankle_moment_y, l_ankle_moment_z];
    r_a_m = [r_ankle_moment_x, r_ankle_moment_y, r_ankle_moment_z];

    %% Accelerations
    LLML_Acc = Marker_ACC(LLML_pos_proc);
    RLML_Acc = Marker_ACC(RLML_pos_proc);
    LLML_Pos = LLML_pos_proc{1,1}(:,1:3);
    RLML_Pos = RLML_pos_proc{1,1}(:,1:3);

    % Synchronize with GRFs
    LLML_Acc = spline_Synch(f1, LLML_Acc);
    RLML_Acc = spline_Synch(f2, RLML_Acc);
    LLML_Pos = spline_Synch(f1, LLML_Pos);
    RLML_Pos = spline_Synch(f2, RLML_Pos);

    % Synchronize joint moments
    left_knee_moment   = spline_Synch(f1, l_k_m);
    right_knee_moment  = spline_Synch(f2, r_k_m);
    left_hip_moment    = spline_Synch(f1, l_h_m);
    right_hip_moment   = spline_Synch(f2, r_h_m);
    left_ankle_moment  = spline_Synch(f1, l_a_m);
    right_ankle_moment = spline_Synch(f2, r_a_m);

end
