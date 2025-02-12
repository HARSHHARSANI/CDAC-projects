# Load cleaned data from HDFS
df_from_hdfs = spark.read.parquet("hdfs://namenode:9000/data/youtube_categories")

# Convert to Pandas for Tableau Export
df_pandas = df_from_hdfs.toPandas()

# Save as CSV for Tableau
df_pandas.to_csv("youtube_categories.csv", index=False)
