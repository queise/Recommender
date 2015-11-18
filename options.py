import sys
import os
from time import time
import itertools
import csv
import math as m
import numpy as np
from numpy.random import seed
import pandas as pd
pd.options.mode.chained_assignment = None
import findspark # alternatively, comment an execute: $SPARK_HOME/bin/spark-submit top5recom.py
findspark.init()
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext() # local mode

# path to the purchases dataset:
purch_file="purchases.csv"
# to make the code verbose or not
verb=False
# set the seed:
myseed = 1313 ; seed(myseed)

############################
# Options for recommender:
############################
# users with a number of purchases deviated more than 7 std from the mean are removed (see prepare_df()):
remove_outliers = True
# only train on 2014 data, but evaluate users of all-time by predicting selection of best-sellers for them:
only_last_year  = True

############################
# Options for evaluation:
############################
# root-mean-square error. Automatically = False for 'baseline' or 'only_custom_BS'
include_RMSE = False
# mean average precision at k=5. Automatically = True for 'baseline' or 'only_custom_BS'
include_MAPK = True
# num_u_toeval in MAPK
num_u_toeval = 1000
