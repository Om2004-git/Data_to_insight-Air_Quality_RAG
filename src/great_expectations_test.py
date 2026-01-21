import pandas as pd
import glob
import great_expectations as ge

# -------------------------------
# Load processed parquet files
# -------------------------------

parquet_files = glob.glob("processed/air_quality_delta/**/*.parquet", recursive=True)

df_list = [pd.read_parquet(pq) for pq in parquet_files]
df = pd.concat(df_list, ignore_index=True)

print("\nLoaded Dataset Shape:", df.shape)
print(df.head())

# -------------------------------
# Convert to GE Dataset
# -------------------------------

ge_df = ge.from_pandas(df)

# -------------------------------
# Data Quality Tests (as per PDF)
# -------------------------------

print("\nRunning Data Quality Tests...\n")

# Not null checks
ge_df.expect_column_values_to_not_be_null("pm25")
ge_df.expect_column_values_to_not_be_null("pm10")
ge_df.expect_column_values_to_not_be_null("no2")

# Range checks
ge_df.expect_column_values_to_be_between("pm25", 0, 1000)
ge_df.expect_column_values_to_be_between("pm10", 0, 1000)
ge_df.expect_column_values_to_be_between("no2", 0, 500)

# Schema checks
ge_df.expect_column_to_exist("country")
ge_df.expect_column_to_exist("date")

# -------------------------------
# Run validation
# -------------------------------

results = ge_df.validate()

if results["success"]:
    print("✅ Data Quality Validation Passed!")
else:
    print("❌ Data Quality Validation Failed!")

# Save report
with open("great_expectations_report.txt", "w") as f:
    f.write(str(results))

print("\nValidation report saved to: great_expectations_report.txt")
