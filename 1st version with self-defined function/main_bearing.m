clc;
clear all;
close all;
data_num=300;
feature_num=3;

% Data loading
load('N09_M07_F10_K002_1.mat');
tra_h=N09_M07_F10_K002_1.Y(7).Data(1:250000);
clear N09_M07_F10_K002_1
load('N09_M07_F10_KA01_1.mat');
tra_o=N09_M07_F10_KA01_1.Y(7).Data(1:250000);
clear N09_M07_F10_KA01_1
load('N09_M07_F10_KI01_1.mat');
tra_i=N09_M07_F10_KI01_1.Y(7).Data(1:250000);
clear N09_M07_F10_KI01_1
load('N09_M07_F10_K001_1.mat');
tes_h=N09_M07_F10_K001_1.Y(7).Data(1:250000);
clear N09_M07_F10_K001_1
load('N09_M07_F10_KA04_1.mat');
tes_o=N09_M07_F10_KA04_1.Y(7).Data(1:250000);
clear N09_M07_F10_KA04_1
load('N09_M07_F10_KI21_1.mat');
tes_i=N09_M07_F10_KI21_1.Y(7).Data(1:250000);
clear N09_M07_F10_KI21_1

% Applying EMD and PCA:
tra_h=preprocessing(tra_h,feature_num);
tra_o=preprocessing(tra_o,feature_num);
tra_i=preprocessing(tra_i,feature_num);
tes_h=preprocessing(tes_h,feature_num);
tes_o=preprocessing(tes_o,feature_num);
tes_i=preprocessing(tes_i,feature_num);

% Standardisation
tra_h=zscore(tra_h);
tra_o=zscore(tra_o);
tra_i=zscore(tra_i);
tes_h=zscore(tes_h);
tes_o=zscore(tes_o);
tes_i=zscore(tes_i);

% Keeping the first data_num pieces of data
tra_h=tra_h(1:data_num,:);
tra_o=tra_o(1:data_num,:);
tra_i=tra_i(1:data_num,:);
tes_h=tra_h(1:data_num,:);
tes_o=tra_o(1:data_num,:);
tes_i=tra_i(1:data_num,:);

trainData_hoi=[tra_h;tra_o;tra_i]';
trainLabel_hoi=[ones(1,length(tra_h)),-ones(1,length(tra_o)+length(tra_i))]';

trainData_oi=[tra_o;tra_i]';
trainLabel_oi=[ones(1,length(tra_o)),-ones(1,length(tra_i))]';

testData=[tes_h;tes_o;tes_i]';
testLabel=[zeros(1,length(tes_h)),ones(1,length(tes_o)),ones(1,length(tes_i))+1]';

% n = length(trainData);
fprintf('Pre-processing completed\n');

%% 1. Hard margin with 'rbf' and 'linear' kernel
tic;
disp('1. Hard Margin:');
% tic;
C=10^6;  % for hard margin

kertype='linear';
marginType='hm';

tic;
disp('For the first classifier "svm_hoi(hard margin)":');
svm_hoi=svmTrain(trainData_hoi,trainLabel_hoi,kertype,C,marginType);
disp('*******************************************************************');
disp('For the second classifier "svm_oi(hard margin)":');
svm_oi=svmTrain(trainData_oi,trainLabel_oi,kertype,C,marginType);
toc;
fprintf('Training process completed\n');

% Applying every classifier to test dataset
result_hoi=svmTest(svm_hoi,testData,testLabel);
result_oi=svmTest(svm_oi,testData,testLabel);
fprintf('Validating process completed\n');

% Classifying
for i=1:length(testLabel)
    if result_hoi.d1Test(i)==1
        testLabel2(i)=0;
    elseif result_oi.d1Test(i)==1
        testLabel2(i)=1;
    else
        testLabel2(i)=2;
    end
end
fprintf('Classifying process completed\n');
toc;

disp('*******************************************************************');
disp('Accuracy:');
% The final accuracy of this approach
testAccuracy = length(find(testLabel'==testLabel2))/length(testLabel);
fprintf('The accuracy of test dataset is%6.2f (for hard margin).\n',testAccuracy);
% The accuracy of classifier "svm_hoi"
test1=testLabel2(1:data_num)';
count=length(find(test1==0));
test2=testLabel2(data_num+1:data_num*3)';
count=count+length(find(test2~=0));
testAccuracy_hoi = count/length(testLabel);
fprintf('The accuracy of classifier "svm_hoi" is%6.2f (for hard margin).\n',testAccuracy_hoi);
% The accuracy of classifier "svm_oi"
test1=result_oi.d1Test(data_num+1:data_num*3);
test2=testLabel(data_num+1:data_num*3);
testAccuracy_oi = length(find(test1'==test2))/length(test2);
fprintf('The accuracy of classifier "svm_oi" is%6.2f (for hard margin).\n',testAccuracy_oi);

disp('-------------------------------------------------------------------------------------------');
%% 2. Soft margin

disp('2. Soft Margin:');
tic;
C=0.1;  % for soft margin
marginType='sm';

kertype='linear';
trainData_hoi_sm=trainData_hoi;
trainLabel_hoi_sm=trainLabel_hoi;
trainData_oi_sm=trainData_oi;
trainLabel_oi_sm=trainLabel_oi;
testData_sm=testData;
testLabel_sm=testLabel;

disp('For the first classifier "svm_hoi(soft margin)":');
svm_hoi_sm=svmTrain(trainData_hoi_sm,trainLabel_hoi_sm,kertype,C,marginType);
disp('*******************************************************************');
disp('For the second classifier "svm_oi(soft margin)":');
svm_oi_sm=svmTrain(trainData_oi_sm,trainLabel_oi_sm,kertype,C,marginType);

% Applying every classifier to test dataset
result_hoi_sm=svmTest(svm_hoi_sm,testData,testLabel);
result_oi_sm=svmTest(svm_oi_sm,testData,testLabel);

for i=1:length(testLabel)
    if result_hoi_sm.d1Test(i)==1
        testLabel2_sm(i)=0;
    elseif result_oi_sm.d1Test(i)==1
        testLabel2_sm(i)=1;
    else
        testLabel2_sm(i)=2;
    end
end

toc;

disp('*******************************************************************');
disp('Accuracy:');
% The final accuracy of this approach
testAccuracy_sm = length(find(testLabel==testLabel2_sm'))/length(testLabel);
fprintf('The accuracy of test dataset is%6.2f (for soft margin).\n',testAccuracy_sm);
% The accuracy of classifier "svm_hoi_sm"
test1=testLabel2_sm(1:data_num)';
count=length(find(test1==0));
test2=testLabel2(data_num+1:data_num*3)';
count=count+length(find(test2~=0));
testAccuracy_hoi_sm = count/length(testLabel);
fprintf('The accuracy of classifier "svm_hoi" is%6.2f (for soft margin).\n',testAccuracy_hoi_sm);
% The accuracy of classifier "svm_oi_sm"
test1=result_oi_sm.d1Test(data_num+1:data_num*3);
test2=testLabel(data_num+1:data_num*3);
testAccuracy_oi_sm = length(find(test1'==test2))/length(test2);
fprintf('The accuracy of classifier "svm_oi" is%6.2f (for soft margin).\n',testAccuracy_oi_sm);

