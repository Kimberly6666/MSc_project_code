clc;
clear all;
close all;
data_num=5000;
thresh=95;

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

% Pre-processing:
% 1. Replacing the abnormalities by the mean:
% tra_h=repabnor(tra_h);
% tra_o=repabnor(tra_o);
% tra_i=repabnor(tra_i);
% tes_h=repabnor(tes_h);
% tes_o=repabnor(tes_o);
% tes_i=repabnor(tes_i);

% Other steps of per-processing:
[tra_h,tes_h,feature_num_h]=preprocessing(tra_h,tes_h,thresh);
[tra_o,tes_o,feature_num_o]=preprocessing(tra_o,tes_o,thresh);
[tra_i,tes_i,feature_num_i]=preprocessing(tra_i,tes_i,thresh);

feature_num=max([feature_num_h feature_num_o feature_num_i]);
clear feature_num_h feature_num_o feature_num_i

%  Keeping the first data_num pieces of data
tra_h=tra_h(1:data_num,1:feature_num);
tra_o=tra_o(1:data_num,1:feature_num);
tra_i=tra_i(1:data_num,1:feature_num);
tes_h=tes_h(1:data_num,1:feature_num);
tes_o=tes_o(1:data_num,1:feature_num);
tes_i=tes_i(1:data_num,1:feature_num);

trainData_hoi=[tra_h;tra_o;tra_i];
trainLabel_hoi=[ones(length(tra_h),1);-ones(length(tra_o)+length(tra_i),1)];

trainData_oi=[tra_o;tra_i];
trainLabel_oi=[ones(length(tra_o),1);-ones(length(tra_i),1)];

testData=[tes_h;tes_o;tes_i];
testLabel=[-ones(length(tes_h),1);zeros(length(tes_o),1);ones(length(tes_i),1)];

fprintf('Pre-processing completed\n');

%% Applying model
tic;
kerType='rbf';

% Design PSOGSA optim option and visualisation:
psogsa_option = struct('Max_iteration',20,'noP',5,'w',2,...
    'wMin',0.4,'wMax',0.9,'G0',1,...
    'popcmax',10^6,'popcmin',10^(-3),'popgmax',10^1,'popgmin',10^(-3));

[bestAccuracy,cost,gamma,trace] = psogsasvm(trainData_hoi,trainLabel_hoi,trainData_oi,trainLabel_oi,testData,testLabel,psogsa_option);

figure;
plot(trace,'r-');
xlabel('Generation iteration');
ylabel('Fitness(1-accuracy)');
title('Fitness curve (PSOGSA)')
grid on;
fprintf('***The best cost and gamma are%6.2f,%6.2f respectively.***\n',cost, gamma);

% Train SVM classifiers:
disp('For the first classifier "svm_hoi":');
SVMModel1=fitcsvm(trainData_hoi,trainLabel_hoi,'KernelFunction',kerType,...
    'KernelScale', 1/sqrt(gamma),'Cost',[0,cost;cost,0], 'Standardize',true);
% CVSVMModel1=crossval(SVMModel1,'KFold',10);
CVSVMModel1 = crossval(SVMModel1);
classLoss1 = kfoldLoss(CVSVMModel1);
% trainAccuracy1=kfoldLoss(CVSVMModel1);
fprintf('Training Accuracy of the 1st classifier is:%6.2f \n',1-classLoss1);

% %Visualisation of 1st classifier
% sv1 = SVMModel1.SupportVectors;
% figure
% gscatter(trainData_hoi(:,1),trainData_hoi(:,2),trainLabel_hoi)
% hold on
% plot(sv1(:,1),sv1(:,2),'ko','MarkerSize',10)
% legend('Healthy data','Unhealthy data','Support Vector')
% hold off

disp('*******************************************************************');
disp('For the second classifier "svm_oi":');
SVMModel2=fitcsvm(trainData_oi,trainLabel_oi,'KernelFunction',kerType,...
    'KernelScale', 1/sqrt(gamma),'Cost',[0,cost;cost,0], 'Standardize',true);
CVSVMModel2=crossval(SVMModel2);
classLoss2=kfoldLoss(CVSVMModel2);
fprintf('Training Accuracy of the 2nd classifier is:%6.2f \n',1-classLoss2);

% % Visualisation of 2nt classifier
% sv2 = SVMModel2.SupportVectors;
% figure
% gscatter(trainData_oi(:,1),trainData_oi(:,2),trainLabel_oi)
% hold on
% plot(sv2(:,1),sv2(:,2),'ko','MarkerSize',10)
% legend('Outter-ring-damaged data','Inner-ring-damaged data','Support Vector')
% hold off

fprintf('Training process completed\n');

%% Applying every classifier to test dataset
% the first column is the negative class posterior probabilities, and 
% the second column is the positive class posterior probabilities corresponding to the new observations.
testLabel1 = predict(SVMModel1,testData);
testLabel2 = predict(SVMModel2,testData);

fprintf('Validating process completed\n');

%% Classifying

for i=1:length(testLabel)
    if testLabel1(i)==1
        preLabel(i)=-1;
    elseif testLabel2(i)==1
        preLabel(i)=0;
    else
        preLabel(i)=1;
    end
end
preLabel=preLabel';
fprintf('Classifying process completed\n');

%% disp('*******************************************************************');
disp('Accuracy:');
% The final accuracy of this approach
testAccuracy = length(find(testLabel==preLabel))/length(testLabel);
fprintf('The accuracy of test dataset is%6.2f.\n',testAccuracy);

% The accuracy of classifier "svm_hoi"
test1=preLabel(1:data_num);
count=length(find(test1==-1));
test1=testLabel(data_num+1:data_num*3);
test2=preLabel(data_num+1:data_num*3);
count=count+length(find(test1==test2));
testAccuracy_hoi = count/length(testLabel);
fprintf('The accuracy of classifier "svm_hoi" is%6.2f.\n',testAccuracy_hoi);

% The accuracy of classifier "svm_oi"
test1=preLabel;
test1(testLabel==-1)=[];
test2=testLabel;
test2(testLabel==-1)=[];
testAccuracy_oi = length(find(test1==test2))/length(test2);
fprintf('The accuracy of classifier "svm_oi" is%6.2f.\n',testAccuracy_oi);

disp('-------------------------------------------------------------------------------------------');

toc;
