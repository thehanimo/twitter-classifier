import numpy as np
import pandas as pd
import pickle

import pyspark
import sparknlp
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Tokenizer, StopWordsRemover, RegexTokenizer
from pyspark.sql.functions import regexp_replace
from nltk.corpus import stopwords
from sparknlp.base import Finisher, DocumentAssembler
from sparknlp.annotator import (Tokenizer, Normalizer,
                                LemmatizerModel, StopWordsCleaner)
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, IDF


# conf = pyspark.SparkConf().set("spark.jars.packages",
# "org.mongodb.spark:mongo-spark-connector_2.11:2.2.0")\
# .getOrCreate()
# Connect MongoDB to Spark
working_directory = 'jars/*'


conf = SparkConf().set("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.1.1")
sc = SparkContext(conf=conf)

my_spark = SparkSession(sc) \
    .builder \
    .appName("myApp").config("spark.mongodb.read.connection.uri", 
    "mongodb://127.0.0.1/twitterdata.labeleddata").getOrCreate()

data = my_spark.read.format("mongodb").option("uri", 
"mongodb://127.0.0.1/twitterdata.labeleddata").option("database", 
"twitterdata").option("collection", "labeleddata").load()

# conf = pyspark.SparkConf().setMaster("local[*]").setAppName("SparkTFIDF")
# sc = pyspark.SparkContext(conf=conf)
# spark = pyspark.sql.SparkSession(sc)

# data = spark.read.format("csv")\
#         .option("inferSchema", "true")\
#         .option("header", "true")\
#         .load("/Users/seungpang/Twitter Categorizer/flask-server/tweets.csv") 

# #Select necessary columns - Tweets, Label
data = data.select("Tweets", "Label")

# #Drop null values
data = data.na.drop()

# #from pyspark.sql.functions import countDistinct
correct_label=["celebrity","crypto","stocks","sports","politics"]
data=data.filter(data.Label.isin(correct_label))

#Split into training and test
(df_train, df_test) = data.randomSplit([0.8, 0.2])

#Regex Tokenize
regexTokenizer = RegexTokenizer(inputCol="Tweets", outputCol="regex", pattern="\\W")
regexTokenized = regexTokenizer.transform(df_train)

#Stop Words Remove
remover = StopWordsRemover(inputCol="regex", outputCol="stop_words")
clean_df = remover.transform(regexTokenized)

#tf-idf vectorizer
hashingTF = HashingTF(inputCol="stop_words", outputCol="rawfeatures",numFeatures=50)
featurizedData = hashingTF.transform(clean_df)

idf = IDF(inputCol="rawfeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
df_train_tfidf = idfModel.transform(featurizedData)

#Index Label
string_indexer = StringIndexer(inputCol='Label', outputCol='Label_Indexed')
string_indexer_model = string_indexer.fit(df_train_tfidf)
df_train_final = string_indexer_model.transform(df_train_tfidf)

#Logistic Regression Model
LR_Model = LogisticRegression(featuresCol=idf.getOutputCol(), labelCol=string_indexer_model.getOutputCol())
lr_model = LR_Model.fit(df_train_final)

# Transform the test set 
df_test_token = regexTokenizer.transform(df_test)
df_test_stopwords = remover.transform(df_test_token)
df_test_tf = hashingTF.transform(df_test_stopwords)
df_test_tfidf = idfModel.transform(df_test_tf)
df_test_final= string_indexer_model.transform(df_test_tfidf)

#Prediction
prediction = lr_model.transform(df_test_final)
prediction = prediction.na.drop()
prediction.select("Tweets", "Label", "Label_Indexed", "probability", "prediction").show(10)

#Accuracy
accuracy = prediction.filter(prediction.Label_Indexed == prediction.prediction).count() / float(prediction.count())
print("Accuracy : ",accuracy)


#Dump Pickle ML Model and TFIDF Vect - Need the right syntax for PySpark 
# with open('model_pkl', 'wb') as files:
#     pickle.dump(lr_model, files)



#Load datasets
#tweet_data = pd.read_csv("/Users/seungpang/Twitter Categorizer/flask-server/tweets.csv")
# train_data = spark.read.format("json")\
#         .option("inferSchema", "true")\
#         .option("header", "true")\
#         .load("/Users/seungpang/Twitter Categorizer/flask-server/train_tweet.json") 

# test_data = spark.read.format("json")\
#         .option("inferSchema", "true")\
#         .option("header", "true")\
#         .load("/Users/seungpang/Twitter Categorizer/flask-server/test_tweet.json") 

#Remove bracket in label_name col
# from pyspark.sql.functions import col, concat_ws
# train_data = train_data.withColumn("label",
#    concat_ws(",",col("label_name")))

# test_data = test_data.withColumn("label",
#    concat_ws(",",col("label_name")))

# #Select necessary columns - label_name, text
# train_data = train_data.select("text", "label")
# test_data = test_data.select("text", "label")

#Drop null values
# train_data = train_data.na.drop()
# test_data = test_data.na.drop()

# # Unique Labels - Train - 470 Test - 442
# # train_data.select(countDistinct("label")).show()
# # test_data.select(countDistinct("label")).show()

# data = spark.read.format("csv")\
#         .option("inferSchema", "true")\
#         .option("header", "true")\
#         .load("/Users/seungpang/Twitter Categorizer/flask-server/tweets.csv") 
