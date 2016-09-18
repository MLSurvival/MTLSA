%% FUNCTION Least_TGL
% L21 Joint Feature Learning with Least Squares Loss.
%
%% OBJECTIVE
% argmin_W { sum_i^t (rho2/2* norm (Y - X * W)^2)
%            + opts.rho_L2/2 * \|W\|_2^2 + rho1 * \|W\|_{2,1} }
%
%% INPUT
% X: n * p - data matrix
% Y: n * k - output matrix
% W_old: p * k - coeffcient matrix
% rho1: dual scaler parameter
% rho2: L21-norm parameter
% optional:
%   opts.rho_L2: L2-norm parameter (default = 0).
%
%% OUTPUT
% W: model: d * t
% funcVal: function value vector.
%
%% RELATED PAPERS
%
%   [1] Evgeniou, A. and Pontil, M. Multi-task feature learning, NIPS 2007.
%   [2] Liu, J. and Ye, J. Efficient L1/Lq Norm Regularization, Technical
%       Report, 2010.
%
%% RELATED package
%  MTLSA
%% Code starts here
function [W, funcVal] = Least_L21_Standard(X, Y, rho1,rho2,W_old,opts)

if nargin <6
    error('\n Inputs: X, Y, rho1, should be specified!\n');
end
X = X';

if nargin <7
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

%task_num  = size (Y,2);
%dimension = size(X, 1);
funcVal = [];

bFlag=0; % this flag tests whether the gradient step only changes a little


Wz= W_old;
Wz_old = W_old;

t = 1;
t_old = 0;

iter = 0;
gamma = 1;
gamma_inc = 2;

while iter < opts.maxIter
    alpha = (t_old - 1) /t;
    
    Ws = (1 + alpha) * Wz - alpha * Wz_old;
    %Ws = sparse(Ws);
    % compute function value and gradients of the search point
    gWs  = gradVal_eval(Ws);
    Fs   = funVal_eval (Ws);
    
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma);
        Wzp=sparse(Wzp);
        Fzp = funVal_eval  (Wzp);
        
        delta_Wzp = Wzp - Ws;
        r_sum = norm(delta_Wzp, 'fro')^2;
        %         Fzp_gamma = Fs + trace(delta_Wzp' * gWs)...
        %             + gamma/2 * norm(delta_Wzp, 'fro')^2;
        Fzp_gamma = Fs + sum(sum(delta_Wzp.* gWs))...
            + gamma/2 * norm(delta_Wzp, 'fro')^2;
        
        if (r_sum <=1e-20)
            bFlag=1; % this shows that, the gradient step makes little improvement
            break;
        end
        
        if (Fzp <= Fzp_gamma)
            break;
        else
            gamma = gamma * gamma_inc;
        end
    end
    
    Wz_old = Wz;
    Wz = Wzp;
    
    funcVal = cat(1, funcVal, Fzp + nonsmooth_eval(Wz, rho1));
    
    if (bFlag)
        % fprintf('\n The program terminates as the gradient step changes the solution very small.');
        break;
    end
    
    % test stop condition.
    switch(opts.tFlag)
        case 0
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <= opts.tol)
                    break;
                end
            end
        case 1
            if iter>=2
                if (abs( funcVal(end) - funcVal(end-1) ) <=...
                        opts.tol* funcVal(end-1))
                    break;
                end
            end
        case 2
            if ( funcVal(end)<= opts.tol)
                break;
            end
        case 3
            if iter>=opts.maxIter
                break;
            end
    end
    
    iter = iter + 1;
    t_old = t;
    t = 0.5 * (1 + (1+ 4 * t^2)^0.5);
    
end

W = Wzp;

% private functions

    function [Wp] = FGLasso_projection (W, lambda )
        % solve it in row wise (L_{2,1} is row coupled).
        % for each row we need to solve the proximal opterator
        % argmin_w { 0.5 \|w - v\|_2^2 + lambda_3 * \|w\|_2 }
        
        nm=sqrt(sum(W.^2,2));
        Wp = bsxfun(@times,max(nm-lambda,0)./nm,W);
    end

% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        grad_W = rho2*X*((X' * W-Y))+ rho_L2 * 2 * W;
    end

% smooth part function value.
    function [funcVal] = funVal_eval (W)
        funcVal = 0.5 *rho2* norm ((X' * W-Y),'fro')^2+rho_L2 * norm(W,'fro')^2;
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = sum(rho_1*sqrt(sum(W.^2,2)));
    end
end
