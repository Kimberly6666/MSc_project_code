function [BestAccuracy,bestc,bestg,ConvergenceCurve] = gsasvm(trainData_hoi,trainLabel_hoi,trainData_oi,trainLabel_oi,testData,testLabel,gsa_option)


%% Parameter Initialization
% Max_iteration: Maximum number of iteration.
% noP: Number of masses.
% G0: Initial gravitational constant.
% popcmax:Max SVM value of c.
% popcmin:Min SVM value of c.
% popgmax:Max SVM value of g(gamma).
% popgmin:Min SVM value of g(gamma).

noP = gsa_option.noP;  
Max_iteration = gsa_option.Max_iteration;  
G0=gsa_option.G0; 
Vcmax = gsa_option.popcmax;
Vcmin = -Vcmax;
Vgmax = gsa_option.popgmax;
Vgmin = -Vgmax;
Dim = 2;

CurrentFitness =zeros(noP,1);
BestAccuracy=inf;
BestMass=zeros(1,Dim);
ConvergenceCurve=zeros(1,Max_iteration); %Convergence vector

% CurrentPosition = rand(noP,Dim); %Postition vector
% velocity = .3*randn(noP,Dim) ; %Velocity vector
for i=1:noP
    % Initilize position and velocity of every partical randomly
    CurrentPosition(i,1) = (gsa_option.popcmax-gsa_option.popcmin)*rand+gsa_option.popcmin;
    CurrentPosition(i,2) = (gsa_option.popgmax-gsa_option.popgmin)*rand+gsa_option.popgmin;
    velocity(i,1)=Vcmax*randn(1,1); % rands function is used in original versioin
    velocity(i,2)=Vgmax*randn(1,1); % rands function is used in original versioin
end

%% Main loop
Iteration = 0 ;
while  ( Iteration < Max_iteration )
    Iteration = Iteration + 1;
    G=G0*exp(-20*Iteration/Max_iteration); % alpha = 20
    force=zeros(noP,Dim);
    mass(noP)=0;
    acceleration=zeros(noP,Dim);

    % Calculate fitness of every mass
    for x=1:noP
        SVMModel1=fitcsvm(trainData_hoi,trainLabel_hoi,'KernelFunction',...
            'rbf','KernelScale', 1/sqrt(CurrentPosition(x,2)),...
            'Cost',[0,CurrentPosition(x,1);CurrentPosition(x,1),0], 'Standardize',true);

        SVMModel2=fitcsvm(trainData_oi,trainLabel_oi,'KernelFunction',...
            'rbf','KernelScale', 1/sqrt(CurrentPosition(x,2)),...
            'Cost',[0,CurrentPosition(x,1);CurrentPosition(x,1),0], 'Standardize',true);

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
        CurrentFitness(x) = 1-testAccuracy; % The smaller fitness is, the higher accuracy
    end

    best=min(CurrentFitness);
    worst=max(CurrentFitness);
    
    % Update best & worst
    if(BestAccuracy>best)
        BestAccuracy=best;
        BestMass=CurrentPosition(i,:);            
    end
    
    % Update mass
    for i=1:noP
        mass(i)=(CurrentFitness(i)-0.99*worst)/(best-worst);    
    end
    
    for i=1:noP
        mass(i)=mass(i)*5/sum(mass);    
    end
    
    % Calculate froces
    for i=1:noP
        for j=1:Dim
            for k=1:noP
                if(CurrentPosition(k,j)~=CurrentPosition(i,j))
                    force(i,j)=force(i,j)+ rand()*G*mass(k)*mass(i)*(CurrentPosition(k,j)-CurrentPosition(i,j))/abs(CurrentPosition(k,j)-CurrentPosition(i,j));

                end
            end
        end
    end
    
    % Update accelerate
    for i=1:noP
           for j=1:Dim
                if(mass(i)~=0)
                    acceleration(i,j)=force(i,j)/mass(i);
                end
           end
    end   
    
    % Update velocity
    for i=1:noP
            for j=1:Dim
                velocity(i,j)=rand()*velocity(i,j)+acceleration(i,j);                                           
            end
            % If cross the border?----
            if velocity(i,1) > Vcmax
                velocity(i,1) = Vcmax;
            elseif velocity(i,1) < Vcmin
                velocity(i,1) = Vcmin;
            end
            if velocity(i,2) > Vgmax
                velocity(i,2) = Vgmax;
            elseif velocity(i,2) < Vgmin
                velocity(i,2) = Vgmin;
            end
            
    end
          
    % Update position                                                         
    CurrentPosition = CurrentPosition + velocity;
    for i=1:noP
            % If cross the border?----
            if CurrentPosition(i,1) > gsa_option.popcmax || CurrentPosition(i,1) < gsa_option.popcmin
                CurrentPosition(i,1) = (gsa_option.popcmax-gsa_option.popcmin)*rand+gsa_option.popcmin;
            end
            
            if CurrentPosition(i,2) > gsa_option.popgmax || CurrentPosition(i,2) < gsa_option.popgmin
                CurrentPosition(i,2) = (gsa_option.popgmax-gsa_option.popgmin)*rand+gsa_option.popgmin;
            end
            
    end

    ConvergenceCurve(1,Iteration)=BestAccuracy; 
    disp(['GSA is training SVM (Iteration = ', num2str(Iteration),' ,best accuracy = ', num2str(1-BestAccuracy),')'])        
 
 end

%% Output
% Output the best parameters:
bestc = BestMass(1);
bestg = BestMass(2);
BestAccuracy = 1-BestAccuracy;
end