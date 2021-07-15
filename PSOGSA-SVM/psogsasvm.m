function [BestAccuracy,bestc,bestg,ConvergenceCurve] = psogsasvm(trainData_hoi,trainLabel_hoi,trainData_oi,trainLabel_oi,testData,testLabel,psogsa_option)


%% Parameter Initialization
% Max_iteration: Maximum number of iteration.
% noP: Number of masses.
% G0: Initial gravitational constant.
% popcmax:Max SVM value of c.
% popcmin:Min SVM value of c.
% popgmax:Max SVM value of g(gamma).
% popgmin:Min SVM value of g(gamma).
% w: Inirtia weight.
% wMax: Max inirtia weight.
% wMin: Min inirtia weight.

noP = psogsa_option.noP; 
Max_iteration = psogsa_option.Max_iteration; 
w=psogsa_option.w;     
wMax=psogsa_option.wMax;     
wMin=psogsa_option.wMin;       
G0=psogsa_option.G0; 

Vcmax = psogsa_option.popcmax;
Vcmin = -Vcmax;
Vgmax = psogsa_option.popgmax;
Vgmin = -Vgmax;
Dim = 2;

%Vectores for saving the location and accuracy of the best mass
gBestScore=inf;
gBest=zeros(1,Dim);
ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector

for i=1:noP
    % Initilize position and velocity of every partical randomly
    CurrentPosition(i,1) = (psogsa_option.popcmax-psogsa_option.popcmin)*rand+psogsa_option.popcmin;
    CurrentPosition(i,2) = (psogsa_option.popgmax-psogsa_option.popgmin)*rand+psogsa_option.popgmin;
    Velocity(i,1)=Vcmax*randn(1,1); % rands function is used in original versioin
    Velocity(i,2)=Vgmax*randn(1,1); % rands function is used in original versioin
end

%% Main loop
Iteration = 0 ;                 
while  ( Iteration < Max_iteration )
    Iteration = Iteration + 1;  
    G=G0*exp(-20*Iteration/Max_iteration);
    force=zeros(noP,Dim);
    mass(noP)=0;
    acceleration=zeros(noP,Dim);
    for i=1:noP
        % Calculate fitness of every mass
        SVMModel1=fitcsvm(trainData_hoi,trainLabel_hoi,'KernelFunction',...
            'rbf','KernelScale', 1/sqrt(CurrentPosition(i,2)),...
            'Cost',[0,CurrentPosition(i,1);CurrentPosition(i,1),0], 'Standardize',true);

        SVMModel2=fitcsvm(trainData_oi,trainLabel_oi,'KernelFunction',...
            'rbf','KernelScale', 1/sqrt(CurrentPosition(i,2)),...
            'Cost',[0,CurrentPosition(i,1);CurrentPosition(i,1),0], 'Standardize',true);

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
        fitness = 1-testAccuracy; % The smaller fitness is, the higher accuracy
        CurrentFitness(i) = fitness;

        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=CurrentPosition(i,:);
        end  
    end
    best=min(CurrentFitness);%Equation (3.10)
    worst=max(CurrentFitness);%Equation (3.11)
    
    % Calculate mass
    for i=1:noP
        mass(i)=(CurrentFitness(i)-0.99*worst)/(best-worst);%Equation (3.9) 
    end
    for i=1:noP
        mass(i)=mass(i)*5/sum(mass);%Equation (3.14)  

    end
    
    %Calculate froces
    for i=1:noP
        for j=1:Dim
            for k=1:noP
                if(CurrentPosition(k,j)~=CurrentPosition(i,j))
                    %Equation (3.5)
                    force(i,j)=force(i,j)+ rand()*G*mass(k)*mass(i)*(CurrentPosition(k,j)-CurrentPosition(i,j))/abs(CurrentPosition(k,j)-CurrentPosition(i,j));

                end
            end
        end
    end
    
    %Calculate acceleration
    for i=1:noP
           for j=1:Dim
                if(mass(i)~=0)
                    acceleration(i,j)=force(i,j)/mass(i);%Equation (3.6)
                end
           end
    end
    
    %Update inertia weight
    w=wMin-Iteration*(wMax-wMin)/Max_iteration;
    
    %Calculate V
    for i=1:noP
            for j=1:Dim
                %Equation (4.1)
                Velocity(i,j)=w*Velocity(i,j)+rand()*acceleration(i,j) + rand()*(gBest(j)-CurrentPosition(i,j));           
            end
            % If velocity cross the border?
            if Velocity(i,1) > Vcmax
                Velocity(i,1) = Vcmax;
            elseif Velocity(i,1) < Vcmin
                Velocity(i,1) = Vcmin;
            end
            if Velocity(i,2) > Vgmax
                Velocity(i,2) = Vgmax;
            elseif Velocity(i,2) < Vgmin            
                Velocity(i,2) = Vgmin;
            end
    end
    
    %Calculate position                                   
    CurrentPosition = CurrentPosition + Velocity ; %Equation (4.2)
    % If position cross the border?----
    for i=1:noP
            if CurrentPosition(i,1) > psogsa_option.popcmax || CurrentPosition(i,1) < psogsa_option.popcmin
                CurrentPosition(i,1) = (psogsa_option.popcmax-psogsa_option.popcmin)*rand+psogsa_option.popcmin;
            end
            
            if CurrentPosition(i,2) > psogsa_option.popgmax || CurrentPosition(i,2) < psogsa_option.popgmin
                CurrentPosition(i,2) = (psogsa_option.popgmax-psogsa_option.popgmin)*rand+psogsa_option.popgmin;
            end
            
    end    
       
    ConvergenceCurve(1,Iteration)=gBestScore; 
    disp(['PSOGSA is training SVM (Iteration = ', num2str(Iteration),' ,accuracy = ', num2str(1-gBestScore),')'])        

end

BestAccuracy = 1-gBestScore;
bestc = gBest(1);
bestg = gBest(2);

end