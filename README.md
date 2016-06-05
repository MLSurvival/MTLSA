Name: MTLSA (A Multi-Task Learning Formulation for Survival Analysis)
Version: 1.0

Together with this Readme.txt are the files mexC.m, MTLSA.m, MTLSA_V2.m survival_data_pre.m, MTLSA.pdf and the folders functions and data.

The folder "data" includes the example data for usage. The original training and testing files are both in ".csv" format. Where each instance is represented as a row in file and the last two columns are survival_times and censored_indicators, respectively. Please refer to “/data/NSBCD_data/NSBCD_train_1.csv” to check detailed format.

Before running the codes, please first run "mexC.m" to mex the related C functions.

The “survival_data_pre.m” is used to generate the target matrix Y, feature matrix X, and the indicator matrix W, from the original training and testing data. Please refer to the Figure 1 in MTLSA.pdf. 
***run example of survival_data_pre.m
>>survival_data_pre 'NSBCD_data/' 'NSBCD_train_1' 'NSBCD_test_1'

The “MTLSA.m” is the implementation of “Multi-Task Learning model for Survival Analysis”, and the result will be saved in the same folder where the training and testing data are saved.  
***run example of MTLSA.m
>>MTLSA 'NSBCD_data/' 'NSBCD_train_1' 'NSBCD_test_1' 100 0.01

The “MTLSA_V2.m” is the implementation of an adaptive variant of MTLSA model, and the result will be saved in the same folder where the training and testing data are saved.  
***run example of MTLSA_V2.m
>>MTLSA_V2 'NSBCD_data/' 'NSBCD_train_1' 'NSBCD_test_1' 100 0.01

You are suggested to read the paper 

Yan Li, Jie Wang, Jieping Ye and Chandan K. Reddy “A Multi-Task Learning
Formulation for Survival Analysis". In Proceedings of the 22nd ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining (KDD'16),
San Francisco, CA, Aug. 2016

which is the “MTLSA.pdf” in this folder for better understanding.

If any problem, please contact Yan Li via yan.li.rock@gmail.com.

Reference Packages:
[1] Liu, Jun, Shuiwang Ji, and Jieping Ye. "SLEP: Sparse learning with efficient projections." Arizona State University 6 (2009): 491.
[2] Zhou, Jiayu, Jianhui Chen, and Jieping Ye. "MALSAR: Multi-task learning via structural regularization." Arizona State University (2011).
