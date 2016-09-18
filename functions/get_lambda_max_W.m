%% Code starts here
function [max_lambda] = get_lambda_max_W(X, Y,rho2,A)
num_task = size(Y,2);
dimension = size(X, 2);
W = zeros(dimension, num_task);

X = X';
rho_L2 = 0;
grad_W = gradVal_eval(W);
max_lambda=0;
for j = 1 : size(grad_W, 1)
    v = grad_W(j, :);
    nm = norm(v, 2);
    if max_lambda < nm
        max_lambda = nm;
    end
end
   
% smooth part gradient.
    function [grad_W] = gradVal_eval(W)
        BX  = X' * W;
        grad_W = [];
        for i = 1:num_task
            %grad_W = cat(2, grad_W, rho2*X*((X' * W(:,i)-Y(:,i))));
             grad_W = cat(2, grad_W, rho2*X*((BX(:,i)-Y(:,i)).*A(:,i)));
        end
        grad_W = grad_W+ rho_L2 * 2 * W;
    end

end