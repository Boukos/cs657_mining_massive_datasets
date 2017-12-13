
# coding: utf-8

# In[137]:


import os
os.listdir(os.curdir)


# In[113]:


#import mllib
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.feature import StandardScalerModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix, BlockMatrix
from pyspark.mllib.linalg import Matrices
from pyspark.mllib.clustering import KMeans, KMeansModel

from math import sqrt
import numpy as np
from numpy import array
import os, csv, sys, time
from random import randint
from itertools import izip, izip_longest


# In[5]:


# def get_input(fn):

fn = "cl_tiny.csv"
cur_dir = os.path.abspath(os.curdir)
input_file_path = os.path.normpath(os.path.join(cur_dir, "..", "data", fn))
print(input_file_path)

os.path.isfile(input_file_path)


# In[6]:


# [(postTitle, postingURL, postLocation, time, lat, long, address, dateRetrieved, post_date, ad), ...]
# tiny input has 30 reviews
raw_ads = sc.textFile(input_file_path)
# set = input.take(3)


# In[7]:


# [ad0, ad1, ..]
ads_rdd = raw_ads.map(lambda x: str(x.decode('utf-8', 'ignore')))
# processed_rdd = input.map(lambda x: str(x.decode('utf-8', 'ignore'))).map(lambda x: x.split(","))
# processed_rdd.take(2)


# In[1]:


# def transpose_rdd()
# tfidf_rdd.flatMap(lambda x: x).take(3)
# flatMap by keeping the column position
flat_rdd = tfidf_rdd.flatMap(lambda row: row.map(lambda col: (col, row.indexOf(col))))
flat_rdd.take(3)
# .map(v => (v._2, v._1)) // key by column position
# .groupByKey.sortByKey   // regroup on column position, thus all elements from the first column will be in the first row
# .map(_._2)              // discard the key, keep only value
# df = rdd.toDF()
# # Grab data from first columns, since it will be transposed to new column headers
# new_header = [i[0] for i in dt.select("_1").rdd.map(tuple).collect()]

# # Remove first column from dataframe
# dt2 = dt.select([c for c in dt.columns if c not in ['_1']])

# # Convert DataFrame to RDD
# rdd = dt2.rdd.map(tuple)

# # Transpose Data
# rddT1 = rdd.zipWithIndex().flatMap(lambda (x,i): [(i,j,e) for (j,e) in enumerate(x)])
# rddT2 = rddT1.map(lambda (i,j,e): (j, (i,e))).groupByKey().sortByKey()
# rddT3 = rddT2.map(lambda (i, x): sorted(list(x), cmp=lambda (i1,e1),(i2,e2) : cmp(i1, i2)))
# rddT4 = rddT3.map(lambda x: map(lambda (i, y): y , x))

# # Convert back to DataFrame (along with header)
# df = rddT4.toDF(new_header)

# return df


# In[8]:


def get_tfidf(rdd):
    # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    # First to compute the IDF vector and second to scale the term frequencies by IDF.
    hashingTF = HashingTF()
    tf = hashingTF.transform(rdd)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf = idf.transform(tf)
    # spark.mllib's IDF implementation provides an option for ignoring terms
    # which occur in less than a minimum number of documents.
    # In such cases, the IDF for these terms is set to 0.
    # This feature can be used by passing the minDocFreq value to the IDF constructor.
    idfIgnore = IDF(minDocFreq=1).fit(tf)
    tfidf_rdd = idfIgnore.transform(tf)
    # rdd of SparseVectors [(doc_id_i: {word_id_j: tfidfscore_j, ...}), ... }]
    # or m docs x n counts
    return tfidf_rdd


# In[ ]:


def get_svd_U(tfidf_rdd, n_topics=3):
    # distributed matrix
    matrix_rdd = RowMatrix(tfidf_rdd)

    matrix_rdd.numRows
    # matrix_rdd.rows.take(3)
    svd = matrix_rdd.computeSVD(3, computeU=True)
    
    # left singular vectors
    # type = RowMatrix
    svd_u = svd.U
    
    # array of DenseVectors, m_documents x n_topics
    # [[topic_i, ...], ...]
    return svd_u.rows.collect()


# In[ ]:


def save_svd_U(svd_U_rdd, fn="svd_u_results.npz"):
    np.savez(fn,np.array(svd_U_rdd))


# In[17]:


tfidf_rdd = get_tfidf(ads_rdd)


# In[81]:


#     sentence = "aa bb ab" * 10 + "a cd " * 10
#     localDoc = [sentence, sentence]
#     doc = sc.parallelize(localDoc).map(lambda line: line.split(" "))
#     model = Word2Vec().setVectorSize(10).setSeed(42).fit(doc)
# i think it is expecting a list of document lists [[word1, word2,...], ...]
def get_word2vec(rdd):
    word2vec = Word2Vec()
    model = word2vec.fit(ads_rdd)



# In[134]:


# k_means
# Build the model (cluster the data)
# kmeans(rdd, k, maxIterations, runs, InitializationMode, seed, initializationSteps, epsilon, initialModel)
def cluster_kmeans_svd(svd_u, k=2, n_iters=10, save_model=False):
    rdd = svd.U.rows
    clusters = KMeans.train(rdd, k, maxIterations=n_iters, initializationMode="random")
    WSSSE = rdd.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))
    if save_model:
        save_cluster_model(clusters, fn)
    
    # do i need to keep the predictions with the keys?
    # returns a list of labels
    return rdd.map(lambda x: ((x[0],x[1]), clusters.predict(x)))


# In[ ]:


def save_cluster_predictions(labels, fn="test_results.pkl"):


# In[111]:


# Evaluate clustering by computing Within Set Sum of Squared Errors
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


# In[135]:


kmeans_mapped = cluster_kmeans_svd(svd.U)


# In[136]:


np.savez("km_cluster_results.npz", kmeans_mapped.collect())


# In[101]:


# Save and load model
def save_cluster_model(clusters, fn="test_model"):
    clusters.save(sc, "target/org/apache/spark/PythonKMeansExample/KMeansModel")
 


# In[105]:


def load_cluster_model(clusters, fn="test_model"):
    sameModel = KMeansModel.load(sc, fn)

