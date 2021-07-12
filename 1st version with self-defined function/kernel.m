function k = kernel(X,Y,type)
switch type
    case 'linear'
        k = X'*Y;
    case 'rbf'
        
        var = std(X(1:end),1)^2; % flag=1 1/N for std (x,flag,dim)  
        for p=1:size(X,2)
            for q=1:size(Y,2)
                k(p,q)=exp((-1/(2*var))*(norm(X(:,p)-Y(:,q))^2)); %Gram Matrix
            end
        end
        
end      
end

