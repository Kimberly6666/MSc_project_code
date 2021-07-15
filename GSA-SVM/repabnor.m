function data=repabnor( dataset )

% 1. Calculating the mean:

% 1.1 Getting the num of missing values and outliers:
% nanLoc=ismissing(dataset);
nanNum=length(find(ismissing(dataset)==1)); % For int, using isnan()
% outlierLoc=isoutlier(dataset,2); % Find the locations of outliers based on the data in each row
outlierNum=length(find(isoutlier(dataset,2)==1));

if nanNum~=0 && outlierNum~=0

    % 1.2 Calculating the sum without missing values and outliers:
    dataset1=dataset.*(~isoutlier(dataset,2));
    % dataset2=fillmissing(dataset1,'constant',0);
    % sumTotal=sum(sum(dataset2(:)));
    sumTotal=sum(sum(dataset1,'omitnan'));
    meanData=sumTotal/(nanNum+outlierNum);

    % 1.3 Filling the missing value
    dataset = fillmissing(dataset,'constant',meanData);

    % 1.4 Replacing the outlier
    dataset(isoutlier(dataset,2))=meanData;
end
data=dataset;
end