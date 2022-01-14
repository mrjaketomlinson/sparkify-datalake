import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, TimestampType as Ts
from pyspark.sql.window import Window
from pyspark.sql.functions import udf, col, row_number
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """
    Creates and returns a spark session with the hadoop-aws package.
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Processes JSON from the song_data udacity S3 bucket and creates two tables --
    songs_table and artists_table -- which are then written as parquet files to
    output_data.
    :spark: Spark session.
    :input_data: String of the S3 bucket path to get the data.
    :output_data: String of the S3 bucket path to output the tables.
    """
    # get filepath to song data file
    song_data = f'{input_data}song_data/'
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select(['song_id', 'title', 'artist_id', 'year', 'duration'])
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write.parquet(f'{output_data}songs_table.parquet')

    # extract columns to create artists table
    artists_table = df.select(['artist_id', 'artist_name', 'artist_location', 'artist_latitude', 'artist_longitude'])

    # write artists table to parquet files
    artists_table.write.parquet(f'{output_data}artists_table.parquet')


def process_log_data(spark, input_data, output_data):
    """
    Processes JSON from the log_data udacity S3 bucket and creates three tables --
    users_table, time_table, and songplays_tabls -- which are then written as
    parquet files to output_data.
    :spark: Spark session.
    :input_data: String of the S3 bucket path to get the data.
    :output_data: String of the S3 bucket path to output the tables.
    """
    # get filepath to log data file
    log_data = f'{input_data}log_data/'

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.where(df.page == 'NextSong')

    # extract columns for users table    
    users_table = df.selectExpr('userId as user_id', 'firstName as first_name', 'lastName as last_name', 'gender', 'level').dropDuplicates()
    
    # write users table to parquet files
    users_table.write.parquet(f'{output_data}users_table.parquet')

    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: x/1000, Int())
    df = df.withColumn('timestamp', get_timestamp(df.ts))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: datetime.fromtimestamp(x), Ts())
    df = df.withColumn('datetime', get_datetime(df.timestamp))
    
    # extract columns to create time table
    time_table = df.select(
        col('datetime').alias('start_time'),
        hour('datetime').alias('hour'),
        dayofmonth('datetime').alias('day'),
        weekofyear('datetime').alias('week'), 
        month('datetime').alias('month'),
        year('datetime').alias('year'),
        date_format('datetime', 'E').alias('weekday')
    )
    
    # write time table to parquet files partitioned by year and month
    time_table.write.partitionBy('year', 'month').parquet(f'{output_data}time_table.parquet')

    # read in song data to use for songplays table
    song_df = spark.read.parquet(f'{output_data}songs_table.parquet')

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = log_df.join(song_df, log_df.song == song_df.title, 'left') \
                            .select(
                                row_number().over(Window.partitionBy().orderBy([log_df.datetime])).alias('songplay_id'), 
                                log_df.datetime.alias('start_time'),
                                log_df.userId.alias('user_id'),
                                log_df.level,
                                song_df.song_id,
                                song_df.artist_id,
                                log_df.sessionId.alias('session_id'),
                                log_df.location,
                                log_df.userAgent.alias('user_agent')
                            )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.partitionBy('year', 'month').parquet(f'{output_data}songplays_table.parquet')


def main():
    """
    1. Creates a spark session.
    2. Processes song data into songs_table and artists table, and writes data
        to S3 in parquet files.
    3. Processes log data into users_table, time_table, and songplays_table, and
        writes data to S3 in parquet files.
    """
    spark = create_spark_session()
    input_data = 's3a://udacity-dend/'
    output_data = 's3a://sparkifydatalake-jacob/'
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
