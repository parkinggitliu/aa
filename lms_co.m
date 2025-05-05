%******************************************************
%  MATLAB Implementation of the LMS Algorithm
%  AUTHOR: Balu Santhanam
%  DATE  : 03/25/03
%  
%  SYNOPSIS : 
%  [filco,y,MSE] = lms_co(u,d,mu,L,init)
%******************************************************
function [filco,y,MSE] = lms_co(u,d,mu,L,init)
if nargin = 4 
   init = zeros(1,L);
elseif nargin < 4 
   error('Insufficient number of input parameters')
end
X = convmtx(u,L).'; % Form Toeplitz convolution matrix
filco(1,:) = init(:).'; n = length(u);
for p = 2:n-L+1
    y(p) = filco(p-1,:)*X(p,:).';
    e(p) = d(p) - y(p);
    filco(p,:) = filco(p-1,:) + mu*e(p)*conj(X(p,:));
    MSE(p) = e(p)^2;
end 
%******************************************************
