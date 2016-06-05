%% FUNCTION Least_L21_Weighted
% L21 Joint Feature Learning with Least Squares Loss of selected label cell.
%
%% OBJECTIVE
% argmin_{B}  1/2 norm(Wo(Y-XB))^2+\frac{\lambda_1}{2} \| B \|_F^2 
%             + \lambda_2 \| B \|_{2,1}
%
%% INPUT
% X: n * p - data matrix
% Y: n * k - output matrix
% W_old: p * k - coeffcient matrix
% rho1: dual scaler parameter.
% A: n * k - weight matrix
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
%% RELATED package
%  MALSAR
%% Code starts here
function [W, funcVal] = Least_L21_Weighted(X, Y, W_old, A, rho1, opts)

if nargin <3
    error('\n Inputs: X, Y, rho1, should be specified!\n');
end
X = X';

if nargin <6
    opts = [];
end

% initialize options.
opts=init_opts(opts);

if isfield(opts, 'rho_L2')
    rho_L2 = opts.rho_L2;
else
    rho_L2 = 0;
end

task_num  = size (Y,2);
dimension = size(X, 1);
num_sample = size(X,2);
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
    
    % compute function value and gradients of the search point
    [gWs,M]  = gradVal_eval(Ws,A);
    Fs   = funVal_eval (Ws,A);
    Ws = (pinv(X))'*M;
    while true
        Wzp = FGLasso_projection(Ws - gWs/gamma, rho1 / gamma);
        Fzp = funVal_eval  (Wzp,A);
        
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
        
        Wp = zeros(size(W));
        
        if opts.pFlag
            parfor i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        else
            for i = 1 : size(W, 1)
                v = W(i, :);
                nm = norm(v, 2);
                if nm == 0
                    w = zeros(size(v));
                else
                    w = max(nm - lambda, 0)/nm * v;
                end
                Wp(i, :) = w';
            end
        end
    end

% smooth part gradient.
    function [grad_W,BX] = gradVal_eval(W,A)
        BX  = X' * W;
            for ii = 1:num_sample
                BX(ii,:)=sequence_bottomup(BX(ii,:),task_num);
            end
        if opts.pFlag
            grad_W = zeros(zeros(W));
            parfor i = 1:task_num
                grad_W (i, :) = X*((BX(:,i)-Y(:,i)).*A(:,i));
            end
        else
            grad_W = [];
            for i = 1:task_num
                grad_W = cat(2, grad_W, X*((BX(:,i)-Y(:,i)).*A(:,i)));
            end
        end
        grad_W = grad_W+ rho_L2 * 2 * W;
    end

% smooth part function value.
    function [funcVal] = funVal_eval (W,A)
        funcVal = 0;
         BX  = X' * W;
            for ii = 1:num_sample
                BX(ii,:)=sequence_bottomup(BX(ii,:),task_num);
            end
        if opts.pFlag
            parfor i = 1: task_num
                funcVal = funcVal + 0.5 * norm ((Y(:,i) - BX(:,i)).*A(:,i))^2;
            end
        else
            for i = 1: task_num
                funcVal = funcVal + 0.5 * norm ((Y(:,i) - BX(:,i)).*A(:,i))^2;
            end
        end
        funcVal = funcVal + rho_L2 * norm(W,'fro')^2;
    end

    function [non_smooth_value] = nonsmooth_eval(W, rho_1)
        non_smooth_value = 0;
        if opts.pFlag
            parfor i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        else
            for i = 1 : size(W, 1)
                w = W(i, :);
                non_smooth_value = non_smooth_value ...
                    + rho_1 * norm(w, 2);
            end
        end
    end
end