function [out ]  = spline_Synch(CTw ,CKs)
% , synch % ,Xa,Xb,Xc ,Ya,Yb,Yc
% This function add points to a curve and synchronize CTw and CKs  
% CTw is the cuve with more data  
Rx = linspace(1,length(CKs) ,length(CKs));
y1 = CKs';
xx1 = linspace(1,length(CKs),length(CTw));
KR1 = spline(Rx,y1,xx1);
%  figure 
% plot(KR1','b'); 
% hold on 
% plot(CTw,'g');

 out = KR1';

