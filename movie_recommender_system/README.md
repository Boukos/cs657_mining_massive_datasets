This program implements a Recommender System to predict ratings of movies, utilizing products of factors for recommendation.  The loss fuction is optimized using ALS.
The best model is found by utilizing 5 fold cross validation that is then used on the "test" set.
The parameters that are searched on are:
	- numBlocks
	- rank or num of latent factors
	- iterations of ALS to run
	- lambda, the regularization parameter in ALS
	- implicitPrefs - explicit or implicit feedback ALS (is the feedback implicit from users?)
	- alpha - implicit feedback parameter, governs baseline confidence 

Steps to run program
 - Download files from the MovieLens website http://files.grouplens.org/datasets/movielens/ml-latest.zip
 - Unzip and put files in data folder
 - Change training file to 1458645.csv (number of rows in file)
 - Update GD/SGD parameters in sgd_lr.py
 - Run . single_sgd_run.sh or call python sgd_lr.py <input file name without extension> <output filename with ext>
 - To run multiple sgd_lr.py against multiple training sizes, use run_sgd.sh
	- The list fns must be file names of files in the data folder (minus the csv extention)

Folder structure:
Pseudo_code - holds project pseudo code
report - contains powerpoint with conclusions and graphs
results - training cross validaiton and test results
src - python code and scripts to run linear regression
src/not_used/ - miscellanous code that was not used for project
test_input - small training sets to test code # not included for submission
data - full data set # not included for submission

