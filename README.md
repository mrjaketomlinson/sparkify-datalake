# Sparkify Data Lake

Sparkify, a pretend startup, wants to analyze data on their music streaming app. What songs are people listening to? What artists get the most play time? What type of users are listening to the most music? In order to answer those questions, this project takes JSON files in S3 buckets, transforms that data, and writes it back to S3 as parquet files. 

## How to use these files
1. Download/Fork the repository.
2. Create a dl.cfg file which includes the following with your own values for the AWS keys:
```
[AWS]
AWS_ACCESS_KEY_ID=YourAccessKey
AWS_SECRET_ACCESS_KEY=YourSecretKey
```
3. In etl.py, modify the output_data variable in the main function to the S3 bucket you'd like to export the data to.
4. Create and SSH into AWS EMR Cluster.
5. Add etl.py and dl.cfg to hadoop directory.
6. Run etl.py (`spark-submit etl.py`). This will process that data listed below and add that data to the S3 bucket you specified.

## Data in this project:
- **song_data** -- A subset of the [Million Song Dataset](http://millionsongdataset.com). Each file contains metadata about a song and the artist of that song.
- **log_data** -- JSON files generated by [eventsim](https://github.com/Interana/eventsim), which simulate activity logs from a music streaming app based on specified configurations.

## Files in this project:
- **etl.py** -- Reads and processes files from the song_data and log_data S3 buckets and loads the processed tables into an output S3 bucket.
- **etl_notebook.ipynb** -- A notebook that walks through the same process as etl.py, but does not write to any output buckets, and uses a subset of the larger data set.