import pandas as pd
import duckdb
import os

# ============================
# Config
# ============================
PARQUET_PATH = "processed/air_quality_delta"
DB_PATH = "data/gold.duckdb"   # keep DB inside data folder

# ============================
# Load Parquet Files + City Partition
# ============================
print("\nLoading Cleaned Parquet Data...")

dataframes = []

for root, dirs, files in os.walk(PARQUET_PATH):
    for file in files:
        if file.endswith(".parquet"):
            file_path = os.path.join(root, file)

            # Extract city from folder name: city=Delhi
            city = None
            for part in file_path.split(os.sep):
                if part.startswith("city="):
                    city = part.replace("city=", "")

            df = pd.read_parquet(file_path)
            df["city"] = city   # add partition column

            dataframes.append(df)

if not dataframes:
    raise Exception("No parquet files found. Run spark_pipeline.py first.")

df = pd.concat(dataframes, ignore_index=True)

print("Dataset Loaded:", df.shape)
print(df.head())

# ============================
# Create DuckDB Gold Database
# ============================
print("\nCreating DuckDB Gold Database...")

con = duckdb.connect(DB_PATH)

# Create Cleaned Gold Table
con.execute("DROP TABLE IF EXISTS air_quality_cleaned")
con.execute("CREATE TABLE air_quality_cleaned AS SELECT * FROM df")

# ============================
# KPI 1: Average Pollution by City
# ============================
kpi_city = con.execute("""
SELECT city,
       ROUND(AVG(pm25),2) AS avg_pm25,
       ROUND(AVG(pm10),2) AS avg_pm10,
       ROUND(AVG(no2),2)  AS avg_no2
FROM air_quality_cleaned
GROUP BY city
ORDER BY avg_pm25 DESC
""").df()

# ============================
# KPI 2: Monthly Pollution Trend
# ============================
kpi_monthly = con.execute("""
SELECT date,
       ROUND(AVG(pm25),2) AS avg_pm25,
       ROUND(AVG(pm10),2) AS avg_pm10,
       ROUND(AVG(no2),2)  AS avg_no2
FROM air_quality_cleaned
GROUP BY date
ORDER BY date
""").df()

# ============================
# KPI 3: Top Polluted Cities
# ============================
kpi_top_polluted = con.execute("""
SELECT city,
       ROUND(AVG(pm25),2) AS avg_pm25
FROM air_quality_cleaned
GROUP BY city
ORDER BY avg_pm25 DESC
LIMIT 3
""").df()

# ============================
# Save Gold KPI Tables
# ============================
con.execute("CREATE OR REPLACE TABLE kpi_city AS SELECT * FROM kpi_city")
con.execute("CREATE OR REPLACE TABLE kpi_monthly AS SELECT * FROM kpi_monthly")
con.execute("CREATE OR REPLACE TABLE kpi_top_polluted AS SELECT * FROM kpi_top_polluted")

# ============================
# Print Results
# ============================
print("\nGold KPIs Generated Successfully!")

print("\nKPI: Average Pollution by City")
print(kpi_city)

print("\nKPI: Monthly Pollution Trend")
print(kpi_monthly)

print("\nKPI: Top Polluted Cities")
print(kpi_top_polluted)

print("\nDuckDB Gold Database Created at:", DB_PATH)
