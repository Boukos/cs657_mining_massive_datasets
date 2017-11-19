#!/usr/bin/python

from pyspark import SparkContext
from pyspark import SparkConf
import itertools
from collections import defaultdict, Counter
import os
import csv
import sys

# dictionary of dictionaries with default value pf 0
stripes_dict = defaultdict(lambda: defaultdict(int))
pair_dict = defaultdict(int)

def get_abs_file_path(file_dir, fn):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.normpath(os.path.join(cur_dir, "..", "..", file_dir, fn))

def save_default_dict_to_disk(count_generator, fn):
    with open(fn, 'wb') as output:
        w = csv.writer(output)
        w.writerow(["m1_m2","counts"])
        for k,v in count_generator:
            key_pair = k[0]
            count = k[1]
            w.writerow([key_pair, count])

def save_rdd_to_disk(output_dir, output_fn, rdd):
    output_path = get_abs_file_path(output_dir, output_fn)
    rdd.saveAsTextFile(output_path)
    #save_default_dict(stripe_count_dict, output_path)
    #print("saving cooccurrence counts")

# stripe is a list of tups, [(m_2, n_2), (m_j, count_j), ...]
def increment_stripes(stripe_x, stripe_y):
    print(stripe_x)
    print(stripe_y)
    x = Counter(stripe_x)
    y = Counter(stripe_y)
    print(x+y)
    return dict(x+y)

# aggregated all the movies reviewed by user u
# movies should be sorted so key pair {(m1, m4) count_1,4} is never (b,a)
# reviewed_movies is list => [m1, m2, ...]
# movies_dict is map of maps =>  {m1:{m2:count2, m4:count4}, ..., m_i:{m_i,j : count_i,j}}
def create_movie_dict(sorted_movies):
    movies_dict = defaultdict(lambda: defaultdict(int))
    for i in range(0, len(sorted_movies)):
        id_i = sorted_movies[i]
        j_start = i + 1
        for j in range(j_start, len(sorted_movies)):
            id_j = sorted_movies[j]
            movies_dict[id_i][id_j] += 1
    return movies_dict

def create_stripe(stripe_dict):
    stripe_list = []
    for movie_i in stripe_dict.keys():
        stripe_list.append((movie_i, stripe_dict[movie_i]))
    return stripe_list

def convert_stripes_to_tups(movie_i, stripe):
    return [((movie_i, movie_j), count) for movie_j, count in stripe.items()]

def main():
    # input parameters
    if len(sys.argv) < 3:
        print("you didnt give directory inputs, using test file")
        input_dir = "test_input"
        input_fn = "ratings_tiny_processed.csv"
        input_file_path = get_abs_file_path(input_dir, input_fn)
        output_fn="test"
    else:
        input_fn = sys.argv[1]
        output_fn = sys.argv[2]
        input_dir = "data"
        input_file_path = get_abs_file_path(input_dir, input_fn)

    # initialize spark
    conf = SparkConf().setMaster("local").setAppName("spark_cooccurrences.py")
    conf.set("spark.executor.memory", "3g")
    conf.set("spark.executor.cores", 2)
    sc = SparkContext(conf = conf)

    # read in file
    data = sc.textFile(input_file_path)

    # take out header
    header = data.first()
    data = data.filter(lambda x: x != header)

    # need to convert list of strings to key value pairs
    #[[u1, mi], ..]
    user_pairs = data.map(lambda x: [int(i) for i in x.split(",")])

    # sorted makes sure that i,j == j,i
    # group pairs [(ui, [sortedmovies_ij]]
    grouped_users = user_pairs.groupByKey().map(lambda x: (x[0], sorted(x[1])))

    # grouped pairs by users and dictionary [(u1, dict1), ..., (ui,dictj)]
    # Using dictionary (stripes) reduces communication costs
    # [(ui, {m_j:{m_k: count_ijk}), ...] count is 1 for all movies
    filtered_movies = grouped_users.map(lambda x: len(x[1]) < 2)
    user_movie_dicts = grouped_users.map(lambda x: (x[0], create_movie_dict(x[1]) ))

    # make key pairs of movie_i, stripe_i
    # [(movie_i, stripe_i), ...]
    movie_stripes = user_movie_dicts.flatMap(lambda x: create_stripe(x[1]))
    print("this is movie_stripes: {}\n-----------\n".format(movie_stripes.collect()))

    # aggregate stripes and sum counts
    # [(m1, {m2:count2, m4:count4}), ...] 
    combined_stripes = movie_stripes.reduceByKey(lambda x,y:
                                              increment_stripes(x,y))
    print("this is combined_stripes: {}\n-----------\n".format(combined_stripes.collect()))

    # convert to pair values and print
    # (mi, mj), count
    counts = combined_stripes.flatMap(lambda x: convert_stripes_to_tups(x[0],
                                                                        x[1]))

    print("this is combined_stripes: {}\n-----------\n".format(counts.collect()))

    # Output results
    output_dir = "output/spark"
    save_rdd_to_disk(output_dir, output_fn, counts)


if __name__ == "__main__":
    main()

def somestuff():
    for k,v in combined_stripes.iteritems():
        print(k)
        print(v)
    #lambda x: [ (m_i, list(x[mi])) for rin x.keys()]
    #movie_stripes = user_movie_dicts.groupByKey(lambda x: [(k,(v )) for k,v in)

    # Count pairs
    stripe_count_rdd = sc.parallelize(((k,v) for k,v in
                                       movie_pairs.countByValue().iteritems()))
    #stripe_count_rdd = sc.parallelize(stripe_count_gen)

