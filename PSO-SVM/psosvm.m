function [bestCVaccuracy,bestc,bestg,fit_gen] = psosvm(trainData_hoi,trainLabel_hoi,trainData_oi,trainLabel_oi,testData,testLabel,pso_option)


%% Parameter Initialization
% c1: Local search ability of PSO.
% c2: Global search ability of PSO.
% Max_iteration: Maximum number of iteration.
% noP: Number of population.
% k:k belongs to [0.1,1.0],the relationship between velocity(V) and position(X)--(V = kX)
% w: Inirtia weight.
% wMax: Max inirtia weight.
% wMin: Min inirtia weight.
% wV:(wV best belongs to [0.8,1.2]), elastic coefficient before velocity
%     when updating velocity
% dt:Elastic coefficient before population
%     when updating population

% popcmax:Max SVM value of c.
% popcmin:Min SVM value of c.
% popgmax:Max SVM value of g(gamma).
% popgmin:Min SVM value of g(gamma).

Vcmax = pso_option.k*pso_option.popcmax;
Vcmin = -Vcmax;
Vgmax = pso_option.k*pso_option.popgmax;
Vgmin = -Vgmax;
%% Initialization
for i=1:pso_option.noP
    % Initilize position and velocity of every partical randomly
    pop(i,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
    pop(i,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
    V(i,1)=Vcmax*rands(1,1);
    V(i,2)=Vgmax*rands(1,1);
    
    % Calculate the initial fitness
    SVMModel1=fitcsvm(trainData_hoi,trainLabel_hoi,'KernelFunction',...
        'rbf','KernelScale', 1/sqrt(pop(i,2)),'Cost',[0,pop(i,1);pop(i,1),0], 'Standardize',true);

    SVMModel2=fitcsvm(trainData_oi,trainLabel_oi,'KernelFunction',...
        'rbf','KernelScale', 1/sqrt(pop(i,2)),'Cost',[0,pop(i,1);pop(i,1),0], 'Standardize',true);
    
    testLabel1 = predict(SVMModel1,testData);
    testLabel2 = predict(SVMModel2,testData);
    
    for y=1:length(testLabel)
        if testLabel1(y)==1
            preLabel(y)=-1;
        elseif testLabel2(y)==1
            preLabel(y)=0;
        else
            preLabel(y)=1;
        end
    end
    
    if size(preLabel,2) ~= 1
        preLabel=preLabel';
    end
    
    testAccuracy = length(find(testLabel==preLabel))/length(testLabel);
    fitness(i) = 1-testAccuracy; % The smaller fitness is, the higher accuracy
    
end

% Find local and global extremas
[global_fitness,bestindex]=min(fitness); % global min
local_fitness=fitness;   % Initialize individual extrema

global_x=pop(bestindex,:);   % global min
local_x=pop;    % Initialize individual extrema

% Every fitness of every population in a certain iteration
avgfitness_gen = zeros(1,pso_option.Max_iteration);

%% Iteration
for i=1:pso_option.Max_iteration
    fprintf('The %d iter: ',i);
    for j=1:pso_option.noP
        
        if i == 1
            wV=pso_option.w;
        else
            %Update the w of PSO
            wV=pso_option.wMin-i*(pso_option.wMax-pso_option.wMin)/pso_option.Max_iteration;
        end
        
        % Update velocity
        V(j,:) = wV*V(j,:) + pso_option.c1*rand*(local_x(j,:) - pop(j,:)) + pso_option.c2*rand*(global_x - pop(j,:));
        % If velocity cross the border?
        if V(j,1) > Vcmax
            V(j,1) = Vcmax;
        end
        if V(j,1) < Vcmin
            V(j,1) = Vcmin;
        end
        if V(j,2) > Vgmax
            V(j,2) = Vgmax;
        end
        if V(j,2) < Vgmin
            V(j,2) = Vgmin;
        end
        
        % Update the population
        pop(j,:)=pop(j,:) + pso_option.dt*V(j,:);
        % If population(position) cross the border?
        if pop(j,1) > pso_option.popcmax
            pop(j,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
        end
        if pop(j,1) < pso_option.popcmin
            pop(j,1) = (pso_option.popcmax-pso_option.popcmin)*rand+pso_option.popcmin;
        end
        if pop(j,2) > pso_option.popgmax
            pop(j,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
        end
        if pop(j,2) < pso_option.popgmin
            pop(j,2) = (pso_option.popgmax-pso_option.popgmin)*rand+pso_option.popgmin;
        end
        
        % Self-adaptive partical variation
        if rand>0.8
            k=ceil(2*rand);
            if k == 1
                pop(j,k) = (pso_option.popcmax-pso_option.popcmin)*rand + pso_option.popcmin;
            end
            if k == 2
                pop(j,k) = (pso_option.popgmax-pso_option.popgmin)*rand + pso_option.popgmin;
            end
        end
        
        % Calculate fitness
        SVMModel1=fitcsvm(trainData_hoi,trainLabel_hoi,'KernelFunction',...
        'rbf','KernelScale', 1/sqrt(pop(j,2)),'Cost',[0,pop(j,1);pop(j,1),0], 'Standardize',true);

        SVMModel2=fitcsvm(trainData_oi,trainLabel_oi,'KernelFunction',...
        'rbf','KernelScale', 1/sqrt(pop(j,2)),'Cost',[0,pop(j,1);pop(j,1),0], 'Standardize',true);
    
        testLabel1 = predict(SVMModel1,testData);
        testLabel2 = predict(SVMModel2,testData);
    
        for z=1:length(testLabel)
            if testLabel1(z)==1
                preLabel(z)=-1;
            elseif testLabel2(z)==1
                preLabel(z)=0;
            else
                preLabel(z)=1;
            end
        end

        if size(preLabel,2) ~= 1
            preLabel=preLabel';
        end

        testAccuracy = length(find(testLabel==preLabel))/length(testLabel);
        fitness(j) = 1-testAccuracy; % The smaller fitness is, the higher accuracy       

        % Update the local best value
        if fitness(j) < local_fitness(j)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        if fitness(j) == local_fitness(j) && pop(j,1) < local_x(j,1)
            local_x(j,:) = pop(j,:);
            local_fitness(j) = fitness(j);
        end
        
        % Update the global best value
        if fitness(j) < global_fitness
            global_x = pop(j,:);
            global_fitness = fitness(j);
        end
    end
    
    fit_gen(i)=global_fitness;
    avgfitness_gen(i) = sum(fitness)/pso_option.noP;
    fprintf('The accuracy is%6.2f.(best:%6.2f) The cost and gamma are%6.2f,%6.2f.\n',...
        1-fitness(j),1-global_fitness,pop(j,1),pop(j,2));
end

%% Output
% Output the best parameters:
bestc = global_x(1);
bestg = global_x(2);
% Output the best accuracy(fitness)
bestCVaccuracy = fit_gen(pso_option.Max_iteration);