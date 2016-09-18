%% file MTLSA_V2.m
% this file shows the usage of Least_L21_Weighted.m function 
% to learn a parth wise solution of MTLSA.V2 
%
%% OBJECTIVE
% argmin_{B}  1/2 norm(Wo(Y-XB))^2+\frac{\lambda_1}{2} \| B \|_F^2 
%             + \lambda_2 \| B \|_{2,1}
%
%% RELATED PAPERS
%  [1]Yan Li, Jie Wang, Jieping Ye and Chandan K. Reddy "A Multi-Task Learning
%     Formulation for Survival Analysis". In Proceedings of the 22nd ACM SIGKDD
%     International Conference on Knowledge Discovery and Data Mining (KDD'16),
%     San Francisco, CA, Aug. 2016
%
%% RELATED PACKAGES 
%  SLEP, MALSAR
%% INPUT
% floder: - the direction where it contains train/test data
% name_train:  - name of training data (.mat is not needed)
% name_test: - name of testing data (.mat is not needed)
% lam_iter: - number of searched lambdas 
% Smallest_lambda_rate: - smallest_lambda/lambda_max, usually set as 0.01
%% Run Example 
%  MTLSA_V2 'NSBCD_data/' 'NSBCD_train_1' 'NSBCD_test_1' 100 0.01


function MTLSA_V2(floder, name_train, name_test,lam_iter,Smallest_lambda_rate)
current_path=cd;
Num_lambda=str2num(lam_iter);
smallest_rate=str2double(Smallest_lambda_rate);
addpath(genpath([current_path '/functions/'])); % load function

% tell the direction where it contains train/test data.
dir=strcat(current_path,'/data/',floder);
load(strcat(dir,name_train,'.mat')); % load training data.
load(strcat(dir,name_test,'.mat')); % load test data.

opts.init = 0;      % guess start point from data. 
opts.tFlag = 1;     % terminate after relative objective value does not changes much.
opts.tol = 10^-4;   % tolerance. 
opts.maxIter = 1000; % maximum iteration number of optimization.


%%build the output matrix
cindex=zeros(Num_lambda, 1);
num_sample = size(Y_test,1);
num_task = size(Y_test,2);

contains=zeros(num_task,1);
AUC_matrix=zeros(Num_lambda,num_task);
dimension=size(X,2);

%% TRAIN
%%Initialize the parameter 
B_old = zeros(dimension, num_task);
%%Calculate the smallest possible \lambad_2 which will make B=0
max_lambda = get_lambda_max_W(X, Y,1,W);
%%pawise wise search for best \lambad_1
lambda = zeros(1,Num_lambda);
for i=1:Num_lambda
    lambda(i)=max_lambda*(smallest_rate)^(i/Num_lambda);
end

ALL_B=cell(1,Num_lambda);
%contains all B's with respect to different lambda
tic
for i = 1: Num_lambda
    if(rem(i,10)==0)
        disp(i);
    end
    %call the FISTA algorithm to solve MTLSA.V2
    [B, funcVal] = Least_L21_Weighted(X, Y, B_old, W, lambda(i), opts);
    % set the solution as the next initial point. 
    % this gives better efficiency. 
    opts.init = 1;
    opts.W0 = B;
    
    B_old=B;
    ALL_B{i}=B;
end
toc; %output the training time

%% TESTING
for i = 1: Num_lambda
    result=X_test*ALL_B{i}; 
     %call the sequence_bottomup function to make sure the prediction 
     %follows the non-negative non-increasing list structure  
    for ii = 1:num_sample
        result(ii,:)=sequence_bottomup(result(ii,:),num_task);
    end
    % evaluate the model performance by concordance index
    cindex(i)=getcindex_nocox(sum(result,2),Time_test,Status_test);
    % evaluate the model performance by calculating AUC for each task
    for k =1:num_task
        temp=find(W_test(:,k));
        label=Y_test(temp,k);
        contains(k)=size(temp,1);
        if length(unique(label))>1
            pred=result(temp,k);
            [X_pred,Y_Pred,T_Pred,AUC_Pred] = perfcurve(label,pred,1);
            AUC_matrix(i,k)=AUC_Pred;
        end
    end
end


%calculating the weighted average of AUC
haveAUC=find(AUC_matrix(1,:));
weighted_AUC=(AUC_matrix(:,haveAUC)*contains(haveAUC,:))/sum(contains(haveAUC));
X_disp = ['Best possible weighted AUC is: ',num2str(max(weighted_AUC)),...
    ' and the Best possible Cindex is: ',num2str(max(cindex))];
disp(X_disp)
disp(['Please check the "',name_test,Smallest_lambda_rate,'_weight_L21_result.mat" file to check all the results']) 
disp('with respect to different lambdas and select the best lambda for your own dataset.')
save(strcat(dir,name_test,Smallest_lambda_rate,'_weight_L21_result.mat'),...
    'ALL_B','weighted_AUC','cindex','lambda','AUC_matrix');
end
