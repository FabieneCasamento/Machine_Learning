%sql
#--- https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-streaming-table
-- Creates a streaming table that processes files stored in the given external location with
-- schema inference and evolution.
> CREATE OR REFRESH STREAMING TABLE raw_data
  AS SELECT * FROM STREAM read_files('abfss://container@storageAccount.dfs.core.windows.net/base/path');

-- Creates a streaming table that processes files with a known schema.
> CREATE OR REFRESH STREAMING TABLE csv_data (
    id int,
    ts timestamp,
    event string
  )
  AS SELECT *
  FROM STREAM read_files(
      's3://bucket/path',
      format => 'csv',
      schema => 'id int, ts timestamp, event string');

-- Stores the data from Kafka in an append-only streaming table.
> CREATE OR REFRESH STREAMING TABLE firehose_raw
  COMMENT 'Stores the raw data from Kafka'
  TBLPROPERTIES ('delta.appendOnly' = 'true')
  AS SELECT
    value raw_data,
    offset,
    timestamp,
    timestampType
  FROM STREAM read_kafka(bootstrapServers => 'ips', subscribe => 'topic_name');

-- Creates a streaming table that scheduled to refresh when upstream data is updated.
-- The refresh frequency of triggered_data is at most once an hour.
> CREATE STREAMING TABLE triggered_data
  TRIGGER ON UPDATE AT MOST EVERY INTERVAL 1 hour
  AS SELECT *
  FROM STREAM source_stream_data;

-- Read data from another streaming table scheduled to run every hour.
> CREATE STREAMING TABLE firehose_bronze
  SCHEDULE EVERY 1 HOUR
  AS SELECT
    from_json(raw_data, 'schema_string') data,
    * EXCEPT (raw_data)
  FROM STREAM firehose_raw;

-- Creates a streaming table with schema evolution and data quality expectations.
-- The table creation or refresh fails if the data doesn't satisfy the expectation.
> CREATE OR REFRESH STREAMING TABLE avro_data (
    CONSTRAINT date_parsing EXPECT (to_date(dt) >= '2000-01-01') ON VIOLATION FAIL UPDATE
  )
  AS SELECT *
  FROM STREAM read_files('gs://my-bucket/avroData');

-- Sets the runtime channel to "PREVIEW"
> CREATE STREAMING TABLE st_preview
  TBLPROPERTIES(pipelines.channel = "PREVIEW")
  AS SELECT * FROM STREAM sales;

-- Creates a streaming table with a column constraint
> CREATE OR REFRESH STREAMING TABLE csv_data (
    id int PRIMARY KEY,
    ts timestamp,
    event string
  )
  AS SELECT *
  FROM STREAM read_files(
      's3://bucket/path',
      format => 'csv',
      schema => 'id int, ts timestamp, event string');

-- Creates a streaming table with a table constraint
> CREATE OR REFRESH STREAMING TABLE csv_data (
    id int,
    ts timestamp,
    event string,
    CONSTRAINT pk_id PRIMARY KEY (id)
  )
  AS SELECT *
  FROM STREAM read_files(
      's3://bucket/path',
      format => 'csv',
      schema => 'id int, ts timestamp, event string');

-- Creates a streaming table with a row filter and a column mask
> CREATE OR REFRESH STREAMING TABLE masked_csv_data (
    id int,
    name string,
    region string,
    ssn string MASK catalog.schema.ssn_mask_fn
  )
  WITH ROW FILTER catalog.schema.us_filter_fn ON (region)
  AS SELECT *
  FROM STREAM read_files('s3://bucket/path/sensitive_data')

-- Creates a streaming table using a FLOW to append data from files
> CREATE OR REFRESH STREAMING TABLE raw_data
  FLOW INSERT BY NAME SELECT * FROM STREAM read_files('abfss://my_path');

-- Creates a streaming table using an AUTO CDC flow to apply changes from a change feed
> CREATE OR REFRESH STREAMING TABLE target
  FLOW AUTO CDC
  FROM stream(cdc_data.users)
  KEYS (userId)
  SEQUENCE BY sequenceNum
  STORED AS SCD TYPE 1;