function data = preprocessing( dataset,feature_num )
% input: row dataset
% output: EMD -> PCA -> return the principle components of which contributes more than 10%

% Applying EMD:
[imf]=emd(dataset,'Interpolation','pchip');
% [imf,residual,info]=emd(dataset,'Interpolation','pchip'); % For analysis
% Plot the result after emd:
% emd(tra_h,'Interpolation','pchip','Display',1) 

% Applying PCA:
[coeff,score,latent] = pca(imf);
% The rows of coeff contain the coefficients for the ten ingredient 
% variables, and its columns correspond to ten principal components.

% % The sum of the latent eigenvalues is unified to 100 to facilitate the observation of the contribution rate
% latent1=100*latent/sum(latent); 
% A=length(latent1);                         % Find out the PC which is bigger than 10%
% data=[];
% for n=1:A
%     if latent1(n)>10
%         data=[data,imf(:,n)];
%     end
% end

data=imf(:,1:feature_num);

% % Plot the result after PCA:
% x_show = {'imf1','imf2','imf3','imf4','imf5','imf6','imf7','imf8','imf9','imf10'};
% pareto(latent1,x_show,1); % The first three PCs contribute 86.7% in total
% pareto(latent2,x_show);
% title('Principle Component on Healthy Data');
% xlabel('Principal Component');
% ylabel('Variance Explained (%)');
% % The lines in the figure indicate the degree to which the cumulative
% % variables are explained:
% print(gcf,'-dpng','PCA.png');

end

