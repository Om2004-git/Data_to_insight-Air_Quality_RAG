import os 
from pyspark.sql import SparkSession #Start Spark Engine 
from pyspark.sql.functions import col, to_date #for col and convert str to date format
from delta import configure_spark_with_delta_pip # enable Delta lake

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "air_quality.csv")

# Configure Spark with Delta Lake
builder = SparkSession.builder \
    .appName("OpenAQ_Delta_ELT") \
    .master("local[*]") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .config("spark.hadoop.native.lib", "false") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs") \
    .config("spark.hadoop.fs.local.block.size", "134217728")
#start Spark Engine
spark = configure_spark_with_delta_pip(builder).getOrCreate()

# Extract
df = spark.read.csv("data/air_quality.csv", header=True, inferSchema=True)
print("Raw Dataset")
df.show()

# cleaning
df_clean = df.dropna()

# transforming
df_clean = df_clean.withColumn("pm25", col("pm25").cast("double"))
df_clean = df_clean.withColumn("pm10", col("pm10").cast("double"))
df_clean = df_clean.withColumn("date", to_date("date"))

print("Cleaned & Transformed Dataset")
df_clean.show()

# SAVE as Delta Lake table (partitioned by country)
df_clean.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("city") \
    .save("processed/air_quality_delta")

print("Delta Lake ELT Pipeline Completed Successfully")
