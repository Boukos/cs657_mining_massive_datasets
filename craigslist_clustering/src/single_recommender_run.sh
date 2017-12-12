#!/bin/bash
fn='reviews'
hdfs_dir='data/'
output_fn=test_results.csv

home_dir=$(eval echo "~$USER")
time_file="$home_dir/cs657/movie_recommender_system/results/time_trials.txt"
time_file2="$home_dir/cs657/movie_recommender_system/results/time_trials2.txt"
hdfs_output="$home_dir/cs657/movie_recommender_system/src/hdfs_stdout.txt"

# might be better to save time results as a var then get append as row to csv
#{ /usr/bin/time -f "%e,%U,%S" spark-submit --master yarn --deploy-mode client movie_recommender.py "$hdfs$fn" $output_fn 2>&1; } 2>> $time_file2 

{ time spark-submit --master yarn --deploy-mode client movie_recommender.py $output_fn 2>&1; } 2>> $time_file2 

#printf "%s,%s\n" $fn >> $time_file

