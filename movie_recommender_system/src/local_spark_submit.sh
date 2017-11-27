\opt\spark\spark-2.2.0-bin-hadoop2.7\bin\spark-submit  \
    -- master local[5] \
    --executor-memory 10g \
    --driver-memory 4g \
    movie_recommender.py
