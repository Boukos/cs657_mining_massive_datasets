Do this assignment in SPARK
• Prepare the data: There are 10 attributes plus the target -trip_duration- (read the description). See if any should be eliminated or transformed. (Use your judgement.)
• Use regresion to train a model that can predict the duration. Do this in the following ways:
	◦ Using a version of regression that optimizes the parameters with Gradient Descent. 
	◦ Using a version of regression that optimizes the parameters with Stochastic Gradient Descent
	◦ Produce a graph of time to train vs. size of the training set to compare both methods
	◦ Add regularization to the model. Try L1 and L2. Compare the results with the previous, unregularized models. 
• For  all the models  you produce above, use a percentage of the training set (80%) to crossvalidate your model (10 folds). Use the rest of the data (20%) as test set. Output the root mean squared error and the mean absolute error. (You cannot use the test data in Kaggle because you do not know the labels.) 
