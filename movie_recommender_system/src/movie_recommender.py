#!/usr/bin/python

#import findspark
#findspark.init()

# import spark stuff
from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import Row

#import mllib
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
#from pyspark.mllib.feature import StandardScalar, StandardScalerModel

# import python stuff
import numpy as np
import os
import csv
import sys
from random import randint
from itertools import izip, izip_longest
import time


# zipped to make rows - take ith element of each list
def train_results_to_disk(fn, zipped_metrics):
    with open(fn, 'a') as results:
        wr = csv.writer(results)
        for row in zipped_metrics:
            wr.writerow(row)

# this is a single row so no need to iterate
def test_results_to_disk(fn, metrics):
    with open(fn, 'a') as results:
        wr = csv.writer(results)
        wr.writerow(metrics)

# helper function to create generators
def create_static_var_gen(var, n_elements):
    return (var for i in xrange(n_elements))

# parse the data, convert str to floats and ints as appropriate
def get_evals(y_and_yhats):
    metrics = RegressionMetrics(y_and_yhats)
    # MSE, RMSE, var, MAE
    return (metrics.meanSquaredError, metrics.rootMeanSquaredError, \
            metrics.explainedVariance, metrics.meanAbsoluteError)

def cv_split(rdd, k_folds, test_k):
    # use mod and rand cv to filter train and validation sets
	# if the row id mod k_folds equals the current validation fold add to validation fold
    train = rdd.filter(lambda x: x[0] % k_folds != test_k)
    test = rdd.filter(lambda x: x[0] % k_folds == test_k)
    return train, test

# result_tups == [(RMSEi, stepi, batch_fraci, reg_typei, reg_paramsi), ... ]
def get_best_params(min_RMSE, zipped_results):
    return [tup[1:] for tup in zipped_results if tup[0] == min_RMSE][0]

def evaluate_recommender(train_set, test_set, rank, itr, reg_param, seed=12345):
    # Evalute the model on training data
    rec_model = ALS.train(train_set, rank=rank, seed=seed, iterations=itr, lambda_=reg_param)
    
    # remove the ratings for the test set
    test_against = test_set.map(lambda x: (x[0], x[1]))
    # make predictions from model
    # returns [((ui, mi), ratingi), ...]
    predictions = rec_model.predictALL(test_against).map(lambda x: ((x[0], x[1]), x[2]))

    # combine on key value pairs of [((ui, mi), (ri, ri_hat)), ...]
    ratings_and_preds = test_set.map(lambda x: (x[0], x[1]), x[2]).join(predictions)
    return get_evals(ratings_and_preds.map(lambda x: x[1]))

def main():
    start_time = time.time()
    # input parameters
    # filenames not given use default
    if len(sys.argv) < 3:
        # assuming hydra hdfs with test_input directory
        print("you didnt give directory inputs, using test file")
        input_dir = "data"
        input_fn = "ratings.csv"
        input_file_path = os.path.join(input_dir, input_fn)
        output_fn="original_test.csv"

    # filenames given, assuming in hydra
    else:
        # expecting full filepath from bash
        input_fn = sys.argv[1]
        output_fn = sys.argv[2]
        input_dir = "data"
        input_file_path = os.path.join(input_dir, input_fn+".csv")
        print("\n________------------________\n")
        print(input_file_path)

    # Optimization params
    # since we have ratings our feedback is explicit, 
    # therefore ignoring the implicit parameters
    ranks = [x for x in range(5, 20, 5)]
    reg_params = [x / float(1000) for x in range(1, 600, 100)]
    iterations = 5
    seed = 12345
    # choose model based on lowest RMSE or MAE
    optimize_RMSE = True

    # CV
    # dataset size is ~26E6
    run_sample_pct = 1E-5
    k_folds = 5

    # param lists
    reg_param_list, ranks_list = [], []
	
	# metric lists
    MSE_results, RMSE_results, exp_vars, MAE_results = [], [], [], []
    MSE_avgs, RMSE_avgs, exp_var_avgs, MAE_avgs= [], [], [], []
    timings = []

    # initialize spark
    conf = SparkConf().setMaster("local").setAppName("linear_regression.py")
    sc = SparkContext(conf = conf)

    # read in data
    all_data = sc.textFile(input_file_path)
    dataset_size = all_data.count() - 1

    # take out header
    header = all_data.first()
    all_data = all_data.filter(lambda x: x != header)

    # take sample of dataset
    data = all_data.sample(withReplacement=False, fraction=run_sample_pct,
                           seed=seed).cache()


    # Need to convert list of strings to mllib.Ratings
    # And add a random fold assignment for CV
    # from -> ["ui, mi, ri, timestamp", ..]
    # to -> [(rand_fold_i, Rating(u1, mi, ri)), ..]
    ratings = data.map(lambda row: row.split(",")) \
                  .map(lambda x: (randint(1, k_folds), Rating(int(x[0]),int(x[1]),float(x[2]))))
    train_set, test_set = ratings.randomSplit([0.8, 0.2], seed=seed)
    train_set.cache()

    # run cross validation on linear regression model
    # SGD step (alpha), batch percent
#--------------------------------Start Grid Search-------------------------------------#
    for rank in ranks:
        for reg_param in reg_params:
            # Build model
            cv_start = time.time()
            for k in range(k_folds):
                #-------------------Start of CV--------------------------#
                # create CV sets
                train_rdd, validate_rdd = cv_split(train_set, k_folds, k)

                # find evaluation metrics
                results = evaluate_recommender(train_rdd, validate_rdd, rank,
                                               iterations, reg_param)

                # store results
                MSE_results.append(results[0])
                RMSE_results.append(results[1])
                exp_vars_results.append(results[2])
                MAE_results.append(results[3])
                #---------------------End of CV--------------------------#

            # update eval lists
            MSE_avgs.append(np.mean(MSE_results))
            RMSE_avgs.append(np.mean(RMSE_results))
            exp_var_avgs.append(np.mean(exp_vars_results))
            MAE_avgs.append(np.mean(MAE_results))

            # reset cv lists
            MSE_results, RMSE_results, exp_vars = [], [], []

            # update param lists
            rank_tracker.append(rank)
            reg_param_tracker.append(reg_param)

            # update timings
            timings.append(time.time() - cv_start)
#--------------------------------End Grid Search-------------------------------------#

    # Finished Grid Search Cross Validation runs
    # save to disk
    fn = os.path.join("..","results","training_results.csv")

    # n length generators for static values
    # e.g. (5,5,5,5,5,...)
    n_runs = len(rank_tracker)
    run_size = create_static_var_gen(run_sample_pct * dataset_size, n_runs)
    iters = create_static_var_gen(iterations, n_runs)
    cv_folds = create_static_var_gen(k_folds, n_runs)
    
	
    # izip returns generator to save space 
    train_results_to_disk(fn, izip(run_size, RMSE_avgs, MSE_avgs, MAE_avgs,
                                   exp_var_avgs, timings, rank_tracker, 
                                   reg_param_tracker, iters, cv_folds))

    # delete result lists to save RAM
    del timings, MSE_avgs, MSE_results, RMSE_results, exp_vars_avgs
    del exp_vars_results, MAE_results

    # delete MAE or RMSE
    if optimize_RMSE: del MAE_avgs 
    else: del RMSE_avgs

    # next find best params
    # Create lazy zip to reduce memory
    # results == [(RMSEi or MAEi, ranki, reg_parami), ... ]
    if optimize_RMSE:
        min_param = min(RMSE_avgs)
        zipped_results = izip(RMSE_avgs, rank_tracker, reg_param_tracker)
    else: #use MAE
        min_param = min(MAE_avgs)
        zipped_results = izip(MAE_avgs, rank_tracker, reg_param_tracker)
	
	# iterate through results and find the params with the min loss
    rank, reg_param = get_best_params(min_param, zipped_results)

    # delete param lists to save RAM
    del rank_tracker, reg_param_tracker
    if optimize_RMSE: del MAE_avgs 
    else: del RMSE_avgs

    # Now run tuned model vs test
    test_start = time.time()
	
	# not sure if mllib caches rdd in the background or not
    test_set.cache()
	
	# return test results
    results = evaluate_recommender(train_set, test_set, rank, iterations, reg_param)
    MSE, RMSE, exp_var, MAE = results
    optimizations = create_static_var_gen(optmize_RMSE, n_runs)

    # save test results to local disk
    fn = os.path.join("..","results", output_fn)
    test_results_to_disk(fn, (run_size, RMSE, MSE, MAE, exp_var, timings, rank,
                              reg_param, optimize_RMSE, time.time() - test_start,
                              time.time() - start_time))

if __name__ == "__main__":
    main()
