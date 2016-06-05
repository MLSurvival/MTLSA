%
% In this file, we mex the C files used in this package.

clear, clc;
current_path=cd;

%% Output infor
%%
fprintf('\n ----------------------------------------------------------------------------');
fprintf('\n The program is mexing the C files. Please wait...');
fprintf('\n If you have problem with mex, you can refer to the help of Matlab.');


% files in the folder order
cd([current_path '/functions/Cfunctions/']);
mex sequence_bottomup.c;
mex sequence_topdown.c;


%% Output infor
%% 
fprintf('\n\n The C files in the folder Cfunctions have been successfully mexed.');
fprintf('\n\n You can now use the functions and run the experiment by using MTLSA.m or MTLSA_V2.m.');
fprintf('\n\n Thanks!');
fprintf('\n ----------------------------------------------------------------------------\n');

cd(current_path);