%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   This function is an implementation of the 
%   NLMS algorithm and returns the tap weights
%
%    AUTHORS  :  Santhanam Balasubramaniam
%    DATE     :  04/19/94
%
%    USAGE    : [filco,y,MSE] = nlms_co(x,d,alpha,beta,ntap)
%    filco    : Filter Coefficients matrix
%    err      : Estimation error
%    MSE      : Mean Square Error (dB)
%    x        : Observation Process (1,N)
%    d        : desired process (1,N)
%    ntap     : Number of taps in the filter
%    alpha    : Step size parameter (0 - 2)
%    beta     : Offset parameter (typ: 0.01)
%    y        : Filtered output
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [filco,y,MSE] = nlms_co(x,d,alpha,beta,ntap)
format long
x = x(:).'; d = d(:).';
N = length(x); X = convmtx(x,ntap).';
W(1,:) = zeros(1,ntap); [M,N] = size(X);
out = zeros(1,N); MSE(1) = 1;
for k = 2:1:M-ntap+1
    mu = alpha / ((X(k,:) * X(k,:).') + beta);
    out(k) = W(k-1,:)*X(k,:).';
    error(k) = d(k) - out(k);
    W(k,:) = W(k-1,:) + mu*conj(X(k,:))*error(k);
    MSE(k) = error(k)^2; 
end
filco = W;
y = out;
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
