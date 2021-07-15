function [tra,tes,feature_num]=preprocessing( tradata,tesdata,tresh )

% 2. Standardisation
[tradata,tramean,trastdev] = zscore(tradata);
tesdata=(tesdata-tramean)./trastdev;
% tesdata=(tesdata-tramean);

% 2. Applying EMD
data=[tradata,tesdata];
[imf,residual,~]=emd(data,'Interpolation','pchip');

imf=[imf,residual];
tradata=imf(1:size(imf,1)/2,:);
tesdata=imf((size(imf,1)/2+1):end,:);

% 3. Applying PCA
[coeff,scoreTrain,~,~,explained,~] = pca(tradata);

sum_explained = 0;
idx = 0;
while sum_explained < tresh
    idx = idx + 1;
    sum_explained = sum_explained + explained(idx);
end

train_mean = mean(tradata);
tra = scoreTrain;%key point
tes = (tesdata - train_mean)*coeff;

% 4. Standardisation
[tra,tramean,trastdev] = zscore(tra);
tes=(tes-tramean)./trastdev;

feature_num=idx;

end

