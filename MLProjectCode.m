%Charlie Barrow's code for Machine learning project (City Uni)

%1)
%K-NN best model (Numneighbors, 10):
%Importing data
data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");
%Checking for missing data
sum(ismissing(data));

%Using feature scaling to standadize the the attributes age and balance in database.
stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;
 
stand_bal = (data.Balance - mean(data.Balance)) / std(data.Balance);
data.Balance = stand_bal;
 
 %Building the classifier for K-NN
classification_model = fitcknn(data,'outcome~Age+Balance', 'NumNeighbors',10);
 
%Creating the test and training sets
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);

cross_validated_model  = crossval(classification_model, 'cvpartition', cv);

%generalisation loss
genError = kfoldLoss(cross_validated_model)
 
% Generating the predictions of the model for the test set
predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1));
 
% Analysing the results of the predictions
results = confusionmat(cross_validated_model.Y(test(cv)), predictions);
 




%2)
%Validation and Evaluation of best K-NN model (Numneighbors, 10):

data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");

%Using feature scaling to standadize the the attributes age and balance in database.  
stand_age = (data.Age - mean(data.Age))/std(data.Age);
data.Age = stand_age; 
 

stand_estimted_bal = (data.Balance - mean(data.Balance))/std(data.Balance);
data.Balance = stand_estimted_bal; 
 
 
 %Building the classifier for K-NN
classification_model = fitcknn(data,'outcome~Age+Balance', 'NumNeighbors',10);
 
 
 %Creating the test and training sets
cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
 
 
cross_validated_model = crossval(classification_model,'cvpartition',cv); 

%generalisation loss (for validation set)
genError = kfoldLoss(cross_validated_model)
 
 
 % Generating the predictions of the model for the test set
Predictions_K_1 = predict(cross_validated_model.Trained{1},data(test(cv,1),1:end-1));
Predictions_K_2 = predict(cross_validated_model.Trained{2},data(test(cv,2),1:end-1));
Predictions_K_3 = predict(cross_validated_model.Trained{3},data(test(cv,3),1:end-1));
Predictions_K_4 = predict(cross_validated_model.Trained{4},data(test(cv,4),1:end-1));
Predictions_K_5 = predict(cross_validated_model.Trained{5},data(test(cv,5),1:end-1));
 
Predictions = kfoldPredict(cross_validated_model); 
 
 % Analysing the results of the predictions
Results_K_1 = confusionmat(cross_validated_model.Y(test(cv,1)),Predictions_K_1);
Results_K_2 = confusionmat(cross_validated_model.Y(test(cv,2)),Predictions_K_2);
Results_K_3 = confusionmat(cross_validated_model.Y(test(cv,3)),Predictions_K_3);
Results_K_4 = confusionmat(cross_validated_model.Y(test(cv,4)),Predictions_K_4);
Results_K_5 = confusionmat(cross_validated_model.Y(test(cv,5)),Predictions_K_5);
 
Results_K = Results_K_1 + Results_K_2 + Results_K_3 + Results_K_4 + Results_K_5; 
 
Results = confusionmat(table2cell(data(:,end)), Predictions);

% Using the confusion matrix and confusionmatstats to analyse the results
% of KNN
cm = confusionchart(Results);
cm.Title = 'Bank Marketing Classification Using KNN';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
 
Evaluation_results = confusionmatStats(table2cell(data(:,end)),Predictions);
 
Evaluation_results.groupOrder
 
Evaluation_results.accuracy

Evaluation_results.precision

Evaluation_results.sensitivity

Evaluation_results.Fscore





%3)
%K-NN best model final test with visualisation (Numneighbors, 10):
data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");

 
stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;
 
stand_bal = (data.Balance - mean(data.Balance)) / std(data.Balance);
data.Balance = stand_bal;
 
 
classification_model = fitcknn(data,'outcome~Age+Balance', 'NumNeighbors',10);
 
 
cv = cvpartition(classification_model.NumObservations, 'HoldOut', 0.2);

cross_validated_model  = crossval(classification_model, 'cvpartition', cv);

genError = kfoldLoss(cross_validated_model)
 
 
predictions = predict(cross_validated_model.Trained{1},data(test(cv),1:end-1));
 
results = confusionmat(cross_validated_model.Y(test(cv)), predictions);
 
%visualisation code to show the test and training data outcomes (yes/no)
labels = unique(data.outcome);
class_name = 'K-Nearest Neighbour  (Training Results)';
 
 
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Bal_ran = min(data.Balance(training(cv)))-1:0.01:max(data.Balance(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range, Bal_ran);
Xgrid = [xx1(:) xx2(:)];
 
predict_meshgrid = predict(cross_validated_model.Trained{1}, Xgrid);
 
gscatter(xx1(:), xx2(:), predict_meshgrid, 'rgb');
 
hold on
 
training_data = data(training(cv),:);
y = ismember(training_data.outcome,labels{1});
 
scatter(training_data.Age(y),training_data.Balance(y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~y),training_data.Balance(~y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
xlabel('Age');
ylabel('Balance');
 
title(class_name);
legend off, axis tight
legend(labels, 'Location', [0.45, 0.01, 0.45, 0.05], 'Orientation', 'Horizontal');
 
 
 
 
labels = unique(data.outcome);
classifier_name = 'K-Nearest Neighbour  (Testing Results)';
 
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Bal_range = min(data.Balance(training(cv)))-1:0.01:max(data.Balance(training(cv)))+1;
 
[xx1, xx2] = meshgrid(Age_range, Bal_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_validated_model.Trained{1},XGrid);
 
figure
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
testing_data =  data(test(cv),:);
Y = ismember(testing_data.outcome,labels{1});
 
scatter(testing_data.Age(Y),testing_data.Balance(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Age(~Y),testing_data.Balance(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
 
xlabel('Age');
ylabel('Balance');
 
title(classifier_name);
legend off, axis tight
 
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');








%4)
%Naïve Bayesian best model ('Distribution', 'kernel'):

%Importing data
data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");

 %Using feature scaling to standadize the the attributes age and balance in database. 
stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;
 
stand_bal = (data.Balance - mean(data.Balance)) / std(data.Balance);
data.Balance = stand_bal;
 
 %Building the classifier for NB
class_model = fitcnb(data, 'outcome~Age+Balance', 'Distribution', 'kernel');
 
 %Creating the test and training sets
cv = cvpartition(class_model.NumObservations, 'HoldOut', 0.2);

cross_val = crossval(class_model, 'cvpartition', cv);

%generalisation loss
genError = kfoldLoss(cross_val)
 
 % Generating the predictions of the model for the test set
predictions = predict(cross_val.Trained{1},data(test(cv),1:end-1));

% Analysing the results of the predictions
results = confusionmat(cross_val.Y(test(cv)), predictions);







%5)
%Evaluation and validation of best Naïve bayes model ('Distribution', 'kernel'): 
data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");
 
 %Using feature scaling to standadize the the attributes age and balance in database. 
stand_age = (data.Age - mean(data.Age))/std(data.Age);
data.Age = stand_age; 
 
stand_estimted_bal = (data.Balance - mean(data.Balance))/std(data.Balance);
data.Balance = stand_estimted_bal; 
 
 
 %Building the classifier for NB
classification_model = fitcnb(data,'outcome~Age+Balance', 'Distribution', 'kernel');
 
 
  %Creating the test and training sets
cv = cvpartition(classification_model.NumObservations, 'KFold', 5);
 
 
cross_validated_model = crossval(classification_model,'cvpartition',cv); 

%generalisation loss (validation)
genError = kfoldLoss(cross_validated_model)
 
 
  % Generating the predictions of the model for the test set
Predictions_K_1 = predict(cross_validated_model.Trained{1},data(test(cv,1),1:end-1));
Predictions_K_2 = predict(cross_validated_model.Trained{2},data(test(cv,2),1:end-1));
Predictions_K_3 = predict(cross_validated_model.Trained{3},data(test(cv,3),1:end-1));
Predictions_K_4 = predict(cross_validated_model.Trained{4},data(test(cv,4),1:end-1));
Predictions_K_5 = predict(cross_validated_model.Trained{5},data(test(cv,5),1:end-1));
 
Predictions = kfoldPredict(cross_validated_model); 
 
 
Results_K_1 = confusionmat(cross_validated_model.Y(test(cv,1)),Predictions_K_1);
Results_K_2 = confusionmat(cross_validated_model.Y(test(cv,2)),Predictions_K_2);
Results_K_3 = confusionmat(cross_validated_model.Y(test(cv,3)),Predictions_K_3);
Results_K_4 = confusionmat(cross_validated_model.Y(test(cv,4)),Predictions_K_4);
Results_K_5 = confusionmat(cross_validated_model.Y(test(cv,5)),Predictions_K_5);
 
Results_K = Results_K_1 + Results_K_2 + Results_K_3 + Results_K_4 + Results_K_5; 
 
Results = confusionmat(table2cell(data(:,end)), Predictions);

% Using the confusion matrix and confusionmatstats to analyse the results
cm = confusionchart(Results);
cm.Title = 'Bank Marketing Classification Using Naive Bayes';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
 
Evaluation_results = confusionmatStats(table2cell(data(:,end)),Predictions);
 
Evaluation_results.groupOrder
 
Evaluation_results.accuracy

Evaluation_results.precision

Evaluation_results.sensitivity

Evaluation_results.Fscore







%6)
%Naïve Bayesian best model ('Distribution', 'kernel') final test and visualisation:

data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");
 

stand_age = (data.Age - mean(data.Age)) / std(data.Age);
data.Age = stand_age;
 
stand_bal = (data.Balance - mean(data.Balance)) / std(data.Balance);
data.Balance = stand_bal;
 
 
class_model = fitcnb(data, 'outcome~Age+Balance', 'Distribution', 'kernel');
 
 
cv = cvpartition(class_model.NumObservations, 'HoldOut', 0.2);

cross_val = crossval(class_model, 'cvpartition', cv);

genError = kfoldLoss(cross_val)
 
 
predictions = predict(cross_val.Trained{1},data(test(cv),1:end-1));
 
results = confusionmat(cross_val.Y(test(cv)), predictions);
 

%visualisation
labels = unique(data.outcome);
class_name = 'Naive Bayesian  (Training Results)';
 
 
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Bal_ran = min(data.Balance(training(cv)))-1:0.01:max(data.Balance(training(cv)))+1;
[xx1, xx2] = meshgrid(Age_range, Bal_ran);
Xgrid = [xx1(:) xx2(:)];
 
predict_meshgrid = predict(cross_val.Trained{1}, Xgrid);
 
gscatter(xx1(:), xx2(:), predict_meshgrid, 'rgb');
 
hold on
 
training_data = data(training(cv),:);
y = ismember(training_data.outcome,labels{1});
 
scatter(training_data.Age(y),training_data.Balance(y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(training_data.Age(~y),training_data.Balance(~y), 'o', 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
xlabel('Age');
ylabel('Balance');
 
title(class_name);
legend off, axis tight
legend(labels, 'Location', [0.45, 0.01, 0.45, 0.05], 'Orientation', 'Horizontal');
 
 
 
 
labels = unique(data.outcome);
classifier_name = 'Naive Bayesian (Testing Results)';
 
Age_range = min(data.Age(training(cv)))-1:0.01:max(data.Age(training(cv)))+1;
Bal_range = min(data.Balance(training(cv)))-1:0.01:max(data.Balance(training(cv)))+1;
 
[xx1, xx2] = meshgrid(Age_range, Bal_range);
XGrid = [xx1(:) xx2(:)];
 
predictions_meshgrid = predict(cross_val.Trained{1},XGrid);
 
figure
 
gscatter(xx1(:), xx2(:), predictions_meshgrid,'rgb');
 
hold on
 
testing_data =  data(test(cv),:);
Y = ismember(testing_data.outcome,labels{1});
 
scatter(testing_data.Age(Y),testing_data.Balance(Y), 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'red');
scatter(testing_data.Age(~Y),testing_data.Balance(~Y) , 'o' , 'MarkerEdgeColor', 'black', 'MarkerFaceColor', 'green');
 
 
xlabel('Age');
ylabel('Balance');
 
title(classifier_name);
legend off, axis tight
 
legend(labels,'Location',[0.45,0.01,0.45,0.05],'Orientation','Horizontal');




%7)
%ROC Curve for Naïve Bayes & KNN testing combined

data = readtable("C:\Users\CHARL\OneDrive\Documents\MATLAB\Machine Learning Final project-Charlie Barrow\bankupdate.csv");
 
 
stand_age = (data.Age - mean(data.Age))/std(data.Age);
data.Age = stand_age; 
 
bal_range = (data.Balance - mean(data.Balance))/std(data.Balance);
data.Balance = bal_range; 
 
classification_model = fitcnb(data,'outcome~Age+Balance', 'Distribution', 'kernel');
 
classification_model_1 = fitcknn(data,'outcome~Age+Balance', 'NumNeighbors',10);
 
cv = cvpartition(classification_model.NumObservations, 'KFold', 5)
 
cv_1 = cvpartition(classification_model_1.NumObservations, 'KFold', 5)
 
testing_indexes = test(cv,1);
 
testing_indexes_1 = test(cv_1,1);
 
cross_validated_model = crossval(classification_model,'cvpartition',cv)
 
cross_validated_model_1 = crossval(classification_model_1,'cvpartition',cv)
 
ACL = cross_validated_model.Y(testing_indexes);
 
ACL_1 = cross_validated_model_1.Y(testing_indexes);
 
[labels, scores] = kfoldPredict(cross_validated_model);
 
[labels, scores] = kfoldPredict(cross_validated_model_1);
 
predicted_labels = labels(testing_indexes);
 
predicted_labels_1 = labels(testing_indexes_1);
 
class_scores = scores(testing_indexes, :);
 
class_scores_1 = scores(testing_indexes_1, :);
 
[X1, Y1, T, AUC] = perfcurve(ACL, class_scores(:,2), 'yes');

AUC
 
[X2, Y2, T, AUC] = perfcurve(ACL_1, class_scores_1(:,2), 'yes');

AUC
 
plot(X1,Y1)
hold on
plot(X2,Y2)
legend('Naive Bayes', 'K-Nearest Neighbour','Location','Best')
xlabel('False positive rate', 'Color','red'); ylabel('True positive rate', 'Color','green');
title('ROC Curves for Naive Bayes Classification and K-Nearest Neighbour', 'Color','blue')

