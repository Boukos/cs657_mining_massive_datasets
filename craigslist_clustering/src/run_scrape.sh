#usr/bin/bash

#test_fns='test_input/thousand_processed.csv'
#hdfs_dir='data/'
#output_fn=test_results.csv

#home_dir=$(eval echo "~$USER")
#time_file="$home_dir/hw3/results/time_trials.txt"
#time_file2="$home_dir/hw3/results/time_trials2.txt"


n_locs=473

# might be better to save time results as a var then get append as row to csv
for i in $(seq 0 $n_locs)
do
	service 
	python cl_scraper.py $i
	sleep 5
	#{ /usr/bin/time -f "%e,%U,%S" spark-submit --master yarn --deploy-mode client code.py "$hdfs$fn" $output_fn 2>&1; } 2>> $time_file2 

done

#run_time=$({ /usr/bin/time -f "%e,%U,%S" sleep 1 2>&1;} )
# { time python $p_code $input_fn $output_fn 2>&1; } 2>> $bash_time_output
