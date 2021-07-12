clc;
clear all;
close all;

% Data loading
dataAll = csvread('iris.csv');
label = dataAll(:,5);
data = dataAll(:,1:4);

% Dividing training set and test set with given testRatio
testRatio = 0.2;

trainIndices = crossvalind('HoldOut', size(data, 1), testRatio);
testIndices = ~trainIndices;

trainData = dataAll(trainIndices, :);

trainData_0 = trainData(find(trainData(:,5) == -1),:);
trainData_0(:,5)=[];
trainData_1 = trainData(find(trainData(:,5) == 0),:);
trainData_1(:,5)=[];
trainData_2 = trainData(find(trainData(:,5) == 1),:);
trainData_2(:,5)=[];

testData = data(testIndices, :);
testLabel = label(testIndices, :);

trainData012=[trainData_0;trainData_1;trainData_2]; % Only 2 classifiers are needed
trainData12=[trainData_1;trainData_2];

trainLabel012=[ones(1,length(trainData_0)),-ones(1,length(trainData_1)+length(trainData_2))]';
trainLabel12=[ones(1,length(trainData_1)),-ones(1,length(trainData_2))]';

fprintf('Pre-processing completed\n');

%% Applying model
kerType='linear';
cost=0.1;

% Train SVM classifiers:
disp('For the first classifier "svm_012":');
SVMModel1=fitcsvm(trainData012,trainLabel012,'KernelFunction',kerType,'Cost',[0,cost;cost,0], 'Standardize',true);
% CVSVMModel1=crossval(SVMModel1,'KFold',10);
CVSVMModel1 = crossval(SVMModel1);
classLoss1 = kfoldLoss(CVSVMModel1);
% trainAccuracy1=kfoldLoss(CVSVMModel1);
fprintf('Training Accuracy of the 1st classifier is:%6.2f \n',1-classLoss1);

% %Visualisation of 1st classifier
% sv1 = SVMModel1.SupportVectors;
% figure
% gscatter(trainData012(:,1),trainData012(:,2),trainLabel012)
% hold on
% plot(sv1(:,1),sv1(:,2),'ko','MarkerSize',10)
% legend('Healthy data','Unhealthy data','Support Vector')
% hold off

disp('*******************************************************************');
disp('For the second classifier "svm_12":');
SVMModel2=fitcsvm(trainData12,trainLabel12,'KernelFunction',kerType,'Cost',[0,cost;cost,0], 'Standardize',true);
CVSVMModel2=crossval(SVMModel2);
classLoss2=kfoldLoss(CVSVMModel2);
fprintf('Training Accuracy of the 2nd classifier is:%6.2f \n',1-classLoss2);

% % Visualisation of 2nt classifier
% sv2 = SVMModel2.SupportVectors;
% figure
% gscatter(trainData12(:,1),trainData12(:,2),trainLabel12)
% hold on
% plot(sv2(:,1),sv2(:,2),'ko','MarkerSize',10)
% legend('Outter-ring-damaged data','Inner-ring-damaged data','Support Vector')
% hold off
fprintf('Training process completed\n');

%% Applying every classifier to test dataset
% the first column is the negative class posterior probabilities, and 
% the second column is the positive class posterior probabilities corresponding to the new observations.
[testLabel1,postProb1] = predict(SVMModel1,testData);
[testLabel2,postProb2] = predict(SVMModel2,testData);

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

%% disp('*******************************************************************');
disp('Accuracy:');
% The final accuracy of this approach
testAccuracy = length(find(testLabel==preLabel'))/length(testLabel);
fprintf('The accuracy of test dataset is%6.2f.\n',testAccuracy);
% The accuracy of classifier "svm_012"
preLabel=preLabel';
test1=preLabel(1:50-length(trainData_0));
count=length(find(test1==-1));
test1=testLabel(50-length(trainData_0)+1:length(testLabel));
test2=preLabel(50-length(trainData_0)+1:length(testLabel));
count=count+length(find(test1==test2));
testAccuracy_012 = count/length(testLabel);
fprintf('The accuracy of classifier "svm_012" is%6.2f.\n',testAccuracy_012);
% The accuracy of classifier "svm_12"
test1=preLabel;
test1(testLabel==-1)=[];
test2=testLabel;
test2(testLabel==-1)=[];
testAccuracy_12 = length(find(test1==test2))/length(test2);
fprintf('The accuracy of classifier "svm_12" is%6.2f.\n',testAccuracy_12);