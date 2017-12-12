This program implements a Recommender System to predict ratings of movies, utilizing products of factors for recommendation.  The loss fuction is optimized using ALS.
The best model is found by utilizing 5 fold cross validation that is then used on the "test" set.
The parameters that are searched on are:
	- rank or num of latent factors
	- lambda, the regularization parameter in ALS

Steps to run program
 - Download files from the MovieLens website http://files.grouplens.org/datasets/movielens/ml-latest.zip
 - Unzip and put files in data folder
 - Run . single_recommender_run.sh 

Folder structure:
Pseudo_code - holds project pseudo code
report - contains powerpoint with project description, pseudo code, and conclusions
results - training cross validaiton and test results
src - python code and scripts to run linear regression
src/not_used/ - miscellanous code that was not used for project
test_input - small training sets to test code # not included for submission
data - full data set # not included for submission
