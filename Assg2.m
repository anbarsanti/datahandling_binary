% Data preparation 
testdata = xlsread('TestData.xlsx');
traindata = xlsread('TrainingData.xlsx');
trainlabel = traindata(:,5)
traindata = traindata(:,1:4);


% calculating the mean ignoring NaN
avg_train = nanmean(traindata);

% Scan the indexes of outliers (NaN) and ..
% Impute its valh mean
listnan = [];
nandetected = zeros(1,2);
[row_data, col_data] = size(traindata)
for i = 1:row_data
    for j = 1:col_data
        x = traindata(i,j);
        if isnan(x)
            traindata(i,j)= avg_train(:,j);
            nandetected(1,1) = i;
            nandetected(1,2) = j;
            listnan = [listnan; nandetected];
        end
        i = i+1;
        j = j+1;
    end
end

% IQR (Interquartile Range)
Q1 = quantile(traindata, 0.25);
Q3 = quantile(traindata,0.75);
IQR = Q3 - Q1;
outliers_IQR = (traindata <(Q1 - 1.5*IQR))| (traindata > (Q3 + 1.5*IQR));

% Scan the indexes of outliers (NaN) and ..
% Impute its value with median
traindata_IQR = traindata;
list_outliers_index = [];
outliers_index = zeros(1,2);
[row_data, col_data] = size(traindata_IQR)
for i = 1:row_data
     for j = 1:col_data
         if outliers_IQR(i,j) == 1
             outliers_index = [i,j]
             list_outliers_index = [list_outliers_index; outliers_index];
             traindata_IQR (i,j)= median(traindata_IQR(:,j));
         end
         i = i+1;
         j = j+1;
     end
 end

% Binary Classification Tree 
% (After Removed outliers data based on IQR)

% Create and Train the Classification Tree
tree_IQR = fitctree(traindata_IQR, trainlabel);

% Visualize the tree
view(tree_IQR, 'Mode', 'graph');

% Make Predictions of TestData
label_IQR = predict(tree_IQR, testdata)
disp(label_IQR')



