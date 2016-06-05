%The ?survival_data_pre.m? is used to generate the target matrix Y, feature matrix X, 
%and the indicator matrix W, from the original training and testing data.
%The original training and testing files are both in ".csv" format.
%Where each instance is represented as a row in file and 
%the last two columns are survival_times and censored_indicators, respectively. 

function survival_data_pre (floder, name_train, name_test)
current_path=cd;
dir=strcat(cd,'/data/',floder);
train=strcat(dir,name_train,'.csv');
test=strcat(dir,name_test,'.csv');
data = csvread(train);
data_test = csvread(test);

% the time intervial can be adjusted here.
data(:,end-1)=fix(data(:,end-1));
data_test(:,end-1)=fix(data_test(:,end-1));
%% for example if the orginal survival time is dayly based, then you can
% devided by 7 to get weekly based time intervial.
%data(:,end-1)=fix(data(:,end-1)/7);
%data_test(:,end-1)=fix(data_test(:,end-1)/7);
%%

max_time=max(max(data(:,end-1)),max(data_test(:,end-1)));
max_time
label=data(:,end-1:end);
X=data(:,1:end-2);
nsample=size(label,1);
W=ones(nsample,max_time);
Y=zeros(nsample,max_time);
for i=1:nsample;
    if label(i,2)==0
        W(i,(label(i,1)+1):end)=0;
        Y(i,(label(i,1)+1):end)=0.5;
    end
    Y(i,1:label(i,1))=1;
end
Time=label(:,1);
Status=label(:,2);
save(strcat(dir, name_train,'.mat'),'X','W','Y','Time','Status');


label_test=data_test(:,end-1:end);
X_test=data_test(:,1:end-2);
nsample_test=size(label_test,1);
W_test=ones(nsample_test,max_time);
Y_test=zeros(nsample_test,max_time);
for i=1:nsample_test;
    if label_test(i,2)==0
        W_test(i,(label_test(i,1)+1):end)=0;
        Y_test(i,(label_test(i,1)+1):end)=0.5;
    end
    Y_test(i,1:label_test(i,1))=1;
end
Time_test=label_test(:,1);
Status_test=label_test(:,2);
save(strcat(dir,name_test,'.mat'),'X_test','W_test','Y_test','Time_test','Status_test');
end