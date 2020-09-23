from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import monotonically_increasing_id


# import re
# import os
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np

def getYear(row):
    textDate = row.split('/')
    year = textDate[2].split(' ')[0]
    return int(year)

def convertRate(row):
    return float(row)

def FormatID(IDnum):
    temp = IDnum
    IDnum = IDnum + 1
    return temp

if __name__ == "__main__":
    SparkContext.setSystemProperty('spark.executor.memory', '5g')
    conf = SparkConf().setAppName("KMeans").setMaster("local[*]")
    sc = SparkContext(conf = conf)
    sc.setLogLevel("ERROR")
    print("_________________________________________________________________________________")
    sc._conf.getAll()
    spark = SparkSession.builder.getOrCreate()

    #declare pyspark udf function
    DateToInteger = udf(getYear, IntegerType())
    RateToInteger = udf(convertRate, FloatType())
    #read in data
    data = spark.read.format('csv').options(header='true').load('lessData.csv')
    #select date and salary rows, convert date to integer year format
    DataForKMeans = data.select(monotonically_increasing_id().alias('id'), DateToInteger("HIRE_DT").alias("Year"), RateToInteger("ANNUAL_RT").alias("Rate"))
    DataForKMeans.printSchema()
    DataForKMeans.write.csv("myData.csv")
    vecAssembler = VectorAssembler(inputCols=["Year", "Rate"], outputCol="features")
    new_df = vecAssembler.transform(DataForKMeans)
    new_df.show()

    # dataToArray = new_df.select(columns: _*).collect.map(_.toSeq)


    # # Trains a k-means model.
    # kmeans = KMeans().setK(2).setSeed(1)
    # model = kmeans.fit(new_df.select('features'))


    cost = np.zeros(20)
    conf = SparkConf().set("spark.python.profile", "true")
    for k in range(2,20):
        kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
        model = kmeans.fit(new_df.sample(False,0.1, seed=42))
        cost[k] = model.computeCost(new_df) # requires Spark 2.0 or later
        print("cost of ", k, " centroids is ", cost[k])
    sc.show_profiles()

    fig, ax = plt.subplots(1,1, figsize =(8,6))
    ax.plot(range(2,20),cost[2:20])
    ax.set_xlabel('k')
    ax.set_ylabel('cost')
    plt.show()
    transformed = model.transform(new_df)
    transformed.show(100)

    sc.stop()
    # Evaluate clustering by computing Silhouette score
    # evaluator = ClusteringEvaluator()
    #
    # silhouette = evaluator.evaluate(predictions)
    # print("Silhouette with squared euclidean distance = " + str(silhouette))
    #
    # # Shows the result.
    # centers = model.clusterCenters()
    # print("Cluster Centers: ")
    # for center in centers:
    #     print(center)
