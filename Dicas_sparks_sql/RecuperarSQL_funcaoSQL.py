# no databricks
# Recuperar dado salvo anteriormente no catalog databricks e no bigquery

%sql

SELECT *
FROM `projeto.dataset.tabela`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR);
-- Substitua '1 HOUR' pelo tempo necessário (ex: 1 DAY, 30 MINUTE)

CREATE OR REPLACE TABLE `projeto.dataset.tabela_restaurada` AS
SELECT *
FROM `projeto.dataset.tabela`
FOR SYSTEM_TIME AS OF TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY);

SELECT *
FROM `seu_projeto.seu_dataset.sua_tabela`
FOR SYSTEM_TIME AS OF TIMESTAMP("2026-02-16 08:00:00 UTC");


CREATE OR REPLACE TABLE `seu_projeto.seu_dataset.tabela_recuperada` AS
SELECT *
FROM `seu_projeto.seu_dataset.sua_tabela`
FOR SYSTEM_TIME AS OF TIMESTAMP("2026-02-17 08:00:00 UTC");


# -------------------------------------------------------

#Criar função no sql
# curso databricks

%sql
DROP FUNCTION IF EXISTS farh_to_cels;

CREATE FUNCTIOIN farh_to_cels(farh DOUBLE)
   RETURNS DOUBLE RETURN ((farh - 32) * 5/9);
   

CREATE OR REPLACE TABLE celsius_sql AS
  SELECT farh_to_cels(temperature_F) as Farh_to_cels_convert FROM device_data;
  
  
%sql
SELECT *
FROM celsius_sql;


