
 function [Mar_acc] = Marker_ACC(Mar) 
%  Mar= LASI_pos_proc;
freqTrX = 120 ;% Hz 
    dt = 1 / freqTrX;
    %% Acc 
    diff1_Mar = diff(Mar{1, 1}(:,1:3)) ./ dt;
    diff2_Mar = diff( diff1_Mar) ./ dt;
    windowSize = 31;
    polynomialOrder = 3;
   Mar_acc= sgolayfilt( diff2_Mar, polynomialOrder, windowSize);
 end 