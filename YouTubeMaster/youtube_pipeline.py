from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
import pandas as pd

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 2, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'youtube_data_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
)

def process_data():
    spark = SparkSession.builder.appName("YouTubeDataCleaning").getOrCreate()
    df = spark.read.json("/path/to/youtube_data.json")

    df_cleaned = df.select("id", "snippet.title").where(df["snippet.assignable"] == True)
    df_cleaned.write.mode("overwrite").parquet("hdfs://namenode:9000/data/youtube_categories")

def export_to_csv():
    spark = SparkSession.builder.appName("YouTubeDataExport").getOrCreate()
    df = spark.read.parquet("hdfs://namenode:9000/data/youtube_categories")
    df_pandas = df.toPandas()
    df_pandas.to_csv("/path/to/youtube_categories.csv", index=False)

extract_data = BashOperator(
    task_id='fetch_data',
    bash_command='curl -o /path/to/youtube_data.json "YOUR_YOUTUBE_API_URL"',
    dag=dag,
)

clean_data = PythonOperator(
    task_id='clean_data',
    python_callable=process_data,
    dag=dag,
)

save_to_hdfs = BashOperator(
    task_id='save_to_hdfs',
    bash_command='hdfs dfs -put -f /path/to/youtube_data.json hdfs://namenode:9000/data/',
    dag=dag,
)

export_csv = PythonOperator(
    task_id='export_csv',
    python_callable=export_to_csv,
    dag=dag,
)

extract_data >> clean_data >> save_to_hdfs >> export_csv
