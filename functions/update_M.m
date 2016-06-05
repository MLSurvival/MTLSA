%% FUNCTION update_M
% update M in the ADMM model
%
%% OBJECTIVE
% argmin_M { sum_i^t (0.5 * norm (W o (Y-M)^2)
%            + rho1 * 0.5* \|M-XB^t+\mu^t \|^2_F }
%
%% INPUT
% X: n * p - data matrix
% Y: n * k - output matrix
% B: p * k - coeffcient matrix
% mu: n * k - dual matrix
% rho1: dual scaler parameter.
% W: n * k - weight matrix
% optional:
%   opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
% M: model: n * k
%% RELATED package
%  SLEP
%% Code starts here
function M= update_M(X, Y, Mold,rho1, B, W, mu, opts)

if nargin <7
    error('\n Inputs: X, Y, rho1,B, W, mu should be specified!\n');
end

if nargin <8
    opts = [];
end

% initialize options.
opts=init_opts(opts);
task_num  = size (Y,2);
num_sample = size(X,1);

S=mu-X*B;
M_pre=(Y.*W-rho1*S)./(W+ones(num_sample,task_num)*rho1);
M=order_projection(M_pre);


% private functions 

    function [Mp] = order_projection (Mbp)
        % argmin_w { 0.5 \|w - v\|_2^2}
        Mp = zeros(size(W));
         for ii = 1:num_sample
             %call the sequence_bottomup function to get 
             %non-negative non-increasing list structure  
                Mp(ii,:)=sequence_bottomup(Mbp(ii,:),task_num);
         end   
    end

end