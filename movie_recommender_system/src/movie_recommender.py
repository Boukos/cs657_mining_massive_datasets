#!/usr/bin/python2

# import findspark
# findspark.init()

# import spark stuff
from pyspark import SparkContext
from pyspark import SparkConf

#import mllib
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# import python stuff
import numpy as np
import os, csv, sys, time
from random import randint
from itertools import izip
from short_user_profile import short_review_list
from long_user_profile import long_review_list

def get_movie_names(fn, sc):
    movies_raw_rdd = sc.TextFile(fn)
    header = movies_raw_rdd.first()
    movies_raw_rdd = movies_raw_rdd.filter(lambda line: line != header)
    # [(movieid, 'movie_name')...]
    movies_rdd = movies_raw_rdd.map(lambda line: line.split(","))\
                               .map(lambda x: (int(x[0]), x[1])).cache()

def train_with_new_user(sc, full_reviews_rdd, rank, reg_param, seed=12345, itr=5, get_small_ratings=True):
    if get_small_ratings:
        new_user = short_review_list
    else:
        new_user = long_review_list
    # [(user_id, movie_id, rating),...]
    new_user_rdd = sc.parallelize(new_user).map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
    ratings_rdd = new_user_rdd.union(full_reviews_rdd)
    updated_model = ALS.train(ratings_rdd, rank=rank, seed=seed, iterations=itr, lambda_=reg_param)
    # list of movie ids that user has rated
    movies_user_rated = map(lambda x: x[1], new_user)

    # userid is 0
    # [(0, movie_id), ...]
    unrated_movies_rdd = ratings_rdd.filter(lambda x: x[0] not in movies_user_rated).map(lambda x: (0, x[0]))

    # predict movie ratings
    user_recommendations_rdd = updated_model.predictAll(unrated_movies_rdd)

    # create key value pairs
    recommendation_rdd = user_recommendations_rdd.map(lambda x: (x.product, x.rating))

    top_movies = recommendation_rdd.takeOrdered(25, key=lambda x: x[1])
    print(top_movies)
    test_results_to_disk("top_movies", top_movies)

def log_output(fn, statement):
    with open(fn, 'a') as log:
        wr = csv.writer(log)
        wr.writerow([statement])

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
    return (metrics.meanSquaredError, metrics.rootMeanSquaredError,
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

def convert_to_rating_rdd(rdd):
    return rdd.map(lambda x: x[1])

def evaluate_recommender(train_set, test_set, rank, itr, reg_param, log_fn, verbose=False, seed=12345):
    train_set.cache()
    test_set.cache()
    # logging statements
    if verbose: log_output(log_fn, "Building ALS Model\n")
    if verbose: log_output(log_fn,"train_set:\n{}\n-".format(train_set.take(3)))
    if verbose: log_output(log_fn, "test_set:\n{}\n--".format(test_set.take(3)))

    # Evalute the model on training data
    # ratings - RDD of Ratings or (userid, productID, rating) tuples
    # rank - Number of features to use (also referred to as the number of latent factors)
    rec_model = ALS.train(train_set, rank=rank, seed=seed, iterations=itr, lambda_=reg_param)
    if verbose: log_output(log_fn, "ALS model:\n{}\n--".format(rec_model))

    # remove the ratings for the test set
    # x_test = [(user_id, movie_id), ...]
    x_test = test_set.map(lambda x: (x[0], x[1]))
    if verbose: log_output(log_fn, "Test pairs:\n{}\n--".format(x_test.take(5)))

    # make predictions for test set from model
    # returns [((ui, mi), r_hati), ...]
    # test_preds = sc.parallelize([(1,2),(1,5),(2,1)])
    # predictions = rec_model.predictAll(test_preds)
    # if verbose: log_output(log_fn, "Predictions test:\n{}\n--".format(predictions.take(5)))

    predictions = rec_model.predictAll(x_test).map(lambda x: ((x[0], x[1]), x[2]))
    if verbose: log_output(log_fn, "Predictions:\n{}\n--".format(predictions.take(5)))

    # combine on key value pairs of [((ui, mi), (ri, ri_hat)), ...]
    ratings_and_preds = test_set.map(lambda x: ((x[0], x[1]), x[2])).join(predictions)
    if verbose: log_output(log_fn,"Rates and Pred:\n{}\n--".format(ratings_and_preds.take(5)))

    # need to calculate y-yhat [(ri, r_hati), ...]
    return get_evals(ratings_and_preds.map(lambda x: x[1]))

def grid_search(cv_rdd, k_folds, ranks, reg_params, iterations, num_of_rows, log_fn, verbose, seed, optimize_RMSE ):
    cv_rdd.cache()
    # param lists
    reg_param_tracker, rank_tracker = [], []
    # metric lists
    MSE_results, RMSE_results, exp_vars_results, MAE_results = [], [], [], []
    MSE_avgs, RMSE_avgs, exp_var_avgs, MAE_avgs= [], [], [], []
    timings = []

    for rank in ranks:
        for reg_param in reg_params:
            # Build model
            cv_start = time.time()
            for k in range(k_folds):
                #-------------------Start of CV--------------------------#
                if verbose: log_output(log_fn, "Start CV")
                # create CV sets
                train_rdd, validate_rdd = cv_split(cv_rdd, k_folds, k)
                if verbose: log_output(log_fn, "prerating:\n{}\n--".format(train_rdd.take(5)))

                # remove random cv assignment
                # from -> [(rand_fold_i, Rating(u1, mi, ri)), ..]
                # to -> [(u1, mi, ri)), ..]
                train_rdd = convert_to_rating_rdd(train_rdd)
                validate_rdd = convert_to_rating_rdd(validate_rdd)
                if verbose: log_output(log_fn, "postrating:\n{}\n--".format(train_rdd.take(5)))

                # find evaluation metrics
                try:
                    results = evaluate_recommender(train_rdd, validate_rdd, rank, iterations,
                                                   reg_param, log_fn, verbose, seed)
                except:
                    e = sys.exc_info()[0]
                    log_output(log_fn, e)
                    raise

                # store results
                MSE_results.append(results[0])
                RMSE_results.append(results[1])
                exp_vars_results.append(results[2])
                MAE_results.append(results[3])
                #---------------------End of CV--------------------------#

            # update eval lists
            if verbose: log_output(log_fn, "Completed CV loop")
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


    if verbose: log_output(log_fn, "Completed Grid search")

    # not sure if mllib caches rdd in the background or not
    cv_rdd.unpersist()

    # Finished Grid Search Cross Validation runs
    # save to disk
    fn = os.path.join("..", "results", "training_results.csv")

    # n length generators for static values
    # e.g. (5,5,5,5,5,...)
    n_runs = len(rank_tracker)
    run_size = create_static_var_gen(num_of_rows, n_runs)
    iters = create_static_var_gen(iterations, n_runs)
    cv_folds = create_static_var_gen(k_folds, n_runs)


    if verbose: log_output(log_fn, "Saving training results to disk")
    # izip returns generator to save space
    train_results_to_disk(fn, izip(run_size, RMSE_avgs, MSE_avgs, MAE_avgs,
                                   exp_var_avgs, timings, rank_tracker,
                                   reg_param_tracker, iters, cv_folds))

    if verbose: log_output(log_fn, "Deleting variables")
    # delete result lists to save RAM
    del timings, MSE_avgs, MSE_results, RMSE_results, exp_var_avgs
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

    if verbose: log_output(log_fn, "Finding best params")
    # iterate through results and find the params with the min loss
    rank, reg_param = get_best_params(min_param, zipped_results)

    return rank, reg_param

def get_inputs():
    # input parameters
    # filenames not given use default
    if len(sys.argv) < 3:
        # assuming hydra hdfs with test_input directory
        print("you didnt give directory inputs, using test file")
        input_dir = "data"
        input_fn = "ratings.csv"
        main_dir = os.path.dirname(os.path.abspath(os.curdir))
        input_file_path = os.path.join(main_dir, input_dir, input_fn)
        # input_file_path = os.path.join(input_dir, input_fn) #hdfs path
        print(input_file_path)
        output_fn="original_test.csv"
        movies_file_path = os.path.join(main_dir, input_dir, "movies.csv")

    # filenames given, assuming in hydra
    else:
        # expecting full filepath from bash
        input_fn = sys.argv[1]
        output_fn = sys.argv[2]
        input_dir = "data"
        input_file_path = os.path.join(input_dir, input_fn+".csv")
        print("\n________------------________\n")
        print(input_file_path)
        movies_file_path = os.path.join(input_dir, "movies.csv")

    return input_file_path, movies_file_path, output_fn

def get_ratings_rdd(all_data, get_sample, run_sample_pct, log_fn, verbose, seed=12345):
    # take out header
    header = all_data.first()
    data = all_data.filter(lambda x: x != header)

    if verbose: log_output(log_fn, "no header:\n{}\n--".format(all_data.take(5)))

    # take sample of dataset
    if get_sample:
        data = data.sample(withReplacement=False, fraction=run_sample_pct,
                           seed=seed).cache()

    if verbose: log_output(log_fn, "this is the data after sampling:\n{}\n--".format(data.take(5)))


    # Need to convert list of strings to mllib.Ratings
    # And add a random fold assignment for CV
    # from -> ["ui, mi, ri, timestamp", ..]
    # to -> [(rand_fold_i, Rating(u1, mi, ri)), ..]
    ratings = data.map(lambda row: row.split(",")) \
        .map(lambda x: Rating(int(x[0]),int(x[1]),float(x[2])))

    if verbose: log_output(log_fn, "these are the ratings:\n{}\n--".format(ratings.take(5)))
    return ratings


def main():
    start_time = time.time()

    # initialize spark
    conf = SparkConf().setMaster("local").setAppName("recommender.py")
    sc = SparkContext(conf = conf)

    # log status
    verbose = False
    log_fn = "hdfs_output.txt"

    # program parameters
    get_sample = False
    run_grid_search = False
    run_test_eval = False
    train_new_user = True

    # dataset size is ~2E7
    run_sample_pct = 1E-1
    k_folds = 5
    seed = 12345
    iterations = 5
    rank = 10
    reg_param = 0.05

    input_file_path, movies_file_path, output_fn = get_inputs()

    # read in data
    all_data = sc.textFile(input_file_path)
    if verbose: log_output(log_fn, "this is the original data:\n{}\n--".format(all_data.take(5)))
    dataset_size = all_data.count() - 1

    ratings = get_ratings_rdd(all_data, get_sample, run_sample_pct, log_fn, verbose, seed)
    #movies_rdd = get_movie_names(movies_file_path)

    # this is a transformation so only acts if grid search or run test eval is run
    train_set, test_set = ratings.randomSplit([0.8, 0.2], seed=seed)

    if run_grid_search:
        reg_params = [x / float(1000) for x in range(1, 600, 100)]
        ranks = [x for x in range(5, 20, 5)]
        iterations = 5
        seed = 12345
        # choose model based on lowest RMSE or MAE
        optimize_RMSE = True
        num_of_rows = run_sample_pct * dataset_size
        # assign random partition to each rdd element for cross validation testing
        cv_rdd = train_set.map(lambda x: (randint(1, k_folds), x))
        rank, reg_param = grid_search(cv_rdd, k_folds, ranks, reg_params, iterations, num_of_rows,
                                      log_fn, verbose, seed, optimize_RMSE)
    if run_test_eval:
        # Now run tuned model vs test
        test_start = time.time()

        if verbose: log_output(log_fn, "Persisting datasets before final run")
        # not sure of mllib caches datasets
        train_set.cache()
        test_set.cache()

        if verbose: log_output(log_fn, "About to run Test run")

        # return test results
        results = evaluate_recommender(train_set, test_set, rank, iterations, reg_param)
        MSE, RMSE, exp_var, MAE = results

        if verbose: log_output(log_fn, "Saving test results to disk")

        # save test results to local disk
        fn = os.path.join("..", "results", output_fn)
        test_results_to_disk(fn, (RMSE, MSE, MAE, exp_var, rank,
                                  reg_param, iterations,
                                  time.time() - test_start,
                                  time.time() - start_time))

        if verbose: log_output(log_fn, "Completed run")

    if train_new_user:
        train_with_new_user(sc, ratings, rank, reg_param, seed=12345, itr=iterations, get_small_ratings=True)

if __name__ == "__main__":
    main()
