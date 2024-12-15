import json
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Load configuration from JSON file
with open('config.json') as config_file:
    config = json.load(config_file)

# Load Data
flights_path = config['flights_path']
airports_path = config['airports_path']

flights_df = spark.read.option('inferSchema', 'true').option('header', 'true').csv(flights_path)
airports_df = spark.read.option('inferSchema', 'true').option('header', 'true').csv(airports_path)

# Feature Selection
flights_features = ["MONTH", "DAY", "DAY_OF_WEEK", "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "SCHEDULED_DEPARTURE", "DISTANCE", "SCHEDULED_ARRIVAL", "ARRIVAL_DELAY"]
flights_df = flights_df.select(flights_features)

airports_features = ["IATA_CODE", "CITY", "STATE", "LATITUDE", "LONGITUDE"]
airports_df = airports_df.select(airports_features)

# Joint Table
airports_origin = airports_df.withColumnRenamed("IATA_CODE", "ORIGIN_AIRPORT") \
                             .withColumnRenamed("CITY", "ORIGIN_CITY") \
                             .withColumnRenamed("STATE", "ORIGIN_STATE") \
                             .withColumnRenamed("LATITUDE", "ORIGIN_LATITUDE") \
                             .withColumnRenamed("LONGITUDE", "ORIGIN_LONGITUDE")

airports_destination = airports_df.withColumnRenamed("IATA_CODE", "DESTINATION_AIRPORT") \
                                  .withColumnRenamed("CITY", "DESTINATION_CITY") \
                                  .withColumnRenamed("STATE", "DESTINATION_STATE") \
                                  .withColumnRenamed("LATITUDE", "DESTINATION_LATITUDE") \
                                  .withColumnRenamed("LONGITUDE", "DESTINATION_LONGITUDE")

flights_df = flights_df.join(airports_origin, "ORIGIN_AIRPORT", "left")
flights_df = flights_df.join(airports_destination, "DESTINATION_AIRPORT", "left")

# Missing Value Check
missing_values = flights_df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in flights_df.columns])

# Missing Value Processing
imputer = Imputer(
    inputCols=["ORIGIN_LATITUDE", "ORIGIN_LONGITUDE", "DESTINATION_LATITUDE", "DESTINATION_LONGITUDE", "ARRIVAL_DELAY"],
    outputCols=["out_ORIGIN_LATITUDE", "out_ORIGIN_LONGITUDE", "out_DESTINATION_LATITUDE", "out_DESTINATION_LONGITUDE", "out_ARRIVAL_DELAY"])

model = imputer.fit(flights_df)
final_df = model.transform(flights_df)

final_df = final_df.drop("ORIGIN_LATITUDE", "ORIGIN_LONGITUDE", "DESTINATION_LATITUDE", "DESTINATION_LONGITUDE", "ARRIVAL_DELAY")

final_df = final_df.dropna(subset=["ORIGIN_CITY", "ORIGIN_STATE", "DESTINATION_CITY", "DESTINATION_STATE"])

final_df = final_df.withColumnRenamed("out_ORIGIN_LATITUDE", "ORIGIN_LATITUDE") \
                   .withColumnRenamed("out_ORIGIN_LONGITUDE", "ORIGIN_LONGITUDE") \
                   .withColumnRenamed("out_DESTINATION_LATITUDE", "DESTINATION_LATITUDE") \
                   .withColumnRenamed("out_DESTINATION_LONGITUDE", "DESTINATION_LONGITUDE") \
                   .withColumnRenamed("out_ARRIVAL_DELAY", "ARRIVAL_DELAY")

# Train Test Split
train_df, test_df = final_df.randomSplit([0.7, 0.3], seed=42)

categorical_features = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ORIGIN_CITY", "ORIGIN_STATE", "DESTINATION_CITY", "DESTINATION_STATE"]
numerical_features = ["MONTH", "DAY", "DAY_OF_WEEK", "SCHEDULED_DEPARTURE", "DISTANCE", "SCHEDULED_ARRIVAL", "ORIGIN_LATITUDE", "ORIGIN_LONGITUDE", "DESTINATION_LATITUDE", "DESTINATION_LONGITUDE"]

# Encode the categorical features
indexers = [StringIndexer(inputCol=feature, outputCol=feature + "_index") for feature in categorical_features]

numerical_assembler = VectorAssembler(inputCols=numerical_features, outputCol="numerical_features_vector")

# Scaling down the numerical features
scaler = StandardScaler(inputCol="numerical_features_vector", outputCol="scaled_num_features")

# Combine different features into a vector 
assembler = VectorAssembler(inputCols=[feature + "_index" for feature in categorical_features] + ["scaled_num_features"], outputCol="features")

# Build a Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="ARRIVAL_DELAY")

# Build ML Pipeline
pipeline = Pipeline(stages=indexers + [numerical_assembler, scaler, assembler, lr])

# Train the pipeline using training data
model = pipeline.fit(train_df)

# Evaluate ML Pipeline

# Training MAE
train_predictions = model.transform(train_df)
evaluator = RegressionEvaluator(labelCol="ARRIVAL_DELAY", predictionCol="prediction", metricName="mae")
train_mae = evaluator.evaluate(train_predictions)
print(f"Training MAE: {train_mae}")

# Testing MAE
test_predictions = model.transform(test_df)
evaluator = RegressionEvaluator(labelCol="ARRIVAL_DELAY", predictionCol="prediction", metricName="mae")
test_mae = evaluator.evaluate(test_predictions)
print(f"Testing MAE: {test_mae}")


