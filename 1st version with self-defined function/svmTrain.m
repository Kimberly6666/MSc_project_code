function [ svm ] = svmTrain( trainData,trainLabel,kertype,c,marginType )

% % Check Mercer Condition for Kernel suitability
 k = kernel(trainData,trainData,kertype);
if min(eigs(k)')>10^(-4)  % returns a vector of the six largest magnitude eigenvalues of matrix k
    disp('Linear kernel is legal! (the Gram matrix is positive semi-definite)');
else
    disp('Linear kernel is not legal! (the Gram matrix is not positive semi-definite)');
end

n = length(trainData);
fprintf('kernel computing completed\n');

% Setting parameters
% Obtaining for a0,i (Quadratic programming)
H = trainLabel.*trainLabel'.*k;
f = -ones(n,1);
A = [];
b = [];
Aeq = trainLabel';
beq = 0;
lb = zeros(n,1);
ub = ones(n,1)*c;
fprintf('Ready to function optimset\n');
% options = optimset('LargeScale','off','MaxIter',1000);
options = optimset('MaxIter',1000);
% Algorithm: 'interior-point-convex'
% Diagonostic: 'off' 
% Display: 'final' (only the final displays is shown)
% MaxIter: 1000
% TolFun(OptimalityTolerance): 1e-8
% TolX(StepTolerance): 1e-12
% TolCon(ConstraintTolerance): 1e-8
% LinearSolver: 'auto' (Using 'sparse' (sparse linear algebra) if H matrix is sparse and 'dense' otherwise)

fprintf('Function optimset completed\n');
a0 = [];
a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
fprintf('Function quadroprog completed\n');

% Obtaining w0
w0 = sum((a.*trainLabel)'.*trainData,2); % w0 = 4 x 1
fprintf('w0 computing completed\n');

% select support vector
threshold = 10^(-4);
idx = find(a>threshold);

switch marginType
    case 'hm'
        s = idx(1);
        % calculate b0
        b0 = 1/trainLabel(s)-w0'*trainData(:,s);
        %caculate Discriminant function
        g = w0'*trainData + b0;
        d1Train = sign(g);
        trainAccuracy = length(find(d1Train' == trainLabel))/length(trainLabel);
        fprintf('The accuracy of training dataset is %f.\n',trainAccuracy);
        
    case 'sm'
        % calculate b0
        b0 = mean(trainLabel(idx)-sum(a.*trainLabel.*k(:,idx))');
        % caculate Discriminant function for training data
        g = sum(a.*trainLabel.*k)+b0;
        d1Train = sign(g);
        trainAccuracy = length(find(d1Train' == trainLabel))/length(trainLabel);
        fprintf('The accuracy of training dataset is %f.\n',trainAccuracy);
end

% svm.a=a(idx);
% svm.Xsv=trainData(:,idx);
% svm.Ysv=trainLabel(idx);
% svm.svnum=length(idx);
svm.w=w0;
svm.b=b0;

end