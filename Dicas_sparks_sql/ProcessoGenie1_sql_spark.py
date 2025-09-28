# Databricks notebook source
# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

spark.sql("CREATE CATALOG IF NOT EXISTS academy")

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW CATALOGS;

# COMMAND ----------

# MAGIC %sql
# MAGIC USE CATALOG workspace;
# MAGIC
# MAGIC CREATE DATABASE IF NOT EXISTS genie_aibi;

# COMMAND ----------

# MAGIC %md
# MAGIC ## quero ler o arquivo stoc_name Schema no Catalog academy

# COMMAND ----------

# List all catalogs to find the correct one
display(spark.sql("SHOW CATALOGS"))

# List all schemas in the 'academy' catalog (if it exists)
display(spark.sql("SHOW SCHEMAS IN academy"))

# COMMAND ----------


# List all tables in the 'academy' schema (in the correct catalog, e.g., 'main')
display(
    spark.sql("SHOW TABLES IN academy.stock_name")
)

# COMMAND ----------



# List all tables in the 'academy' schema (in the correct catalog, e.g., 'main')
#display(spark.sql("SHOW TABLES IN academy.stock_name.stock_bigtech"))

# Once you confirm the correct catalog and schema, use the fully qualified table name
df = spark.read.table("academy.stock_name.stock_bigtech")
display(df)

# COMMAND ----------

# Install Prophet if not already installed
%pip install prophet

from prophet import Prophet
import pandas as pd

# Load your data from Spark into a Pandas DataFrame
df = spark.table("academy.stock_name.stock_bigtech").toPandas()

# Prepare data for Prophet
df_prophet = df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'})

# Fit the model
model = Prophet()
model.fit(df_prophet)

# Create future dataframe
future = model.make_future_dataframe(periods=365)  # Forecast 1 year ahead

# Make predictions
forecast = model.predict(future)

# Display forecast
display(pd.concat([df_prophet, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]], axis=1))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT *
# MAGIC FROM ai_forecast(
# MAGIC   TABLE (academy.stock_name.stock_bigtech),
# MAGIC   '2025-12-31',
# MAGIC   'date',
# MAGIC   'close'
# MAGIC )

# COMMAND ----------

forecast_df = spark.sql(
    """
    SELECT *
    FROM ai_forecast(
      TABLE (academy.stock_name.stock_bigtech),
      '2025-12-31',
      'date',
      'close'
    )
    """
)
display(forecast_df)

# COMMAND ----------

forecast_df = spark.sql("""
SELECT *
FROM ai_forecast(
  TABLE academy.stock_name.stock_bigtech,
  '2025-12-31',                -- horizon (ajuste conforme necessário)
  'date',                      -- time_col (ajuste conforme o nome da coluna de data)
  'close'                      -- value_col
)
""")
display(forecast_df)

# COMMAND ----------

