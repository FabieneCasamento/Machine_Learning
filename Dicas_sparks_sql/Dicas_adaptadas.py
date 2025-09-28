# Databricks notebook source
# Dicas ----------------
# <img src="https://raw.githubusercontent.com/Databricks-BR/lab_sql/main/images/header_notebook.png">

#  ===========================----------

# Dicas ----------------
# ### Controle de Alteração de Versões
#
# | versão | data | autor | e-mail | alterações |
# | --- | --- | --- | --- | --- |
# | 1.0 | 20-JUN-2023 | Luis Assunção | luis.assuncao@databricks.com | Primeira versão (criação de massa de dados) |

#  ===========================----------

# Dicas ----------------
# ### Descrição
#
# | projeto | aplicação | módulo | tabela | objetivo |
# | --- | --- | --- | --- | --- |
# | ACADEMY | Laboratório 2 | ETL Bronze | Diversas CSV | Ingestão de arquivos publicos CSV - bases de teste para o Treinamento de SQL |

#  ===========================----------

# Dicas ----------------
# ### Referências
# * [Leitura de Arquivos CSV](https://learn.microsoft.com/pt-br/azure/databricks/external-data/csv)
# * [Notebook Exemplo - CSV](https://docs.databricks.com/_extras/notebooks/source/read-csv-files.html)
# * [Salvando uma Tabela DELTA](https://docs.databricks.com/delta/tutorial.html#create-a-table)

#  ===========================----------

# Dicas ----------------
# ### Parâmetros Iniciais

#  ===========================----------

import pandas as pd
from pyspark.sql import SparkSession

url = f"https://raw.githubusercontent.com//Databricks-BR/lab_sql/main/dados/"
#catalog_name = f"academy"
catalog_name = f"workspace"

prefix_table = f"bronze"

#  ===========================----------

# DBTITLE 1,ALTERE ESSE PARAMETRO
#schema_name  = f"<<<<<-----COLOQUE SEU USER NAME AQUI --------->>>>"
schema_name  = f"Fabiene"

#  ===========================----------


entity_name  = f"dolar"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

print('table_name', table_name)
print('flie_name',file_name)

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - dolar

entity_name  = f"dolar"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - CNAE

entity_name  = f"cnae"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - empresas

entity_name  = f"empresas"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - estabelecimentos

entity_name  = f"estabelecimentos"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - municipios

entity_name  = f"municipios"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------

# DBTITLE 1,Gravando a tabela DELTA - naturezas

entity_name  = f"naturezas"

table_name   = f"{catalog_name}.{schema_name}.{prefix_table}_{entity_name}"
file_name = f"{url}{entity_name}.csv"

df = pd.read_csv(file_name)                          # leitura arquivo CSV utilizando Dataframe Pandas
s_df = spark.createDataFrame(df)                     # converte Dataframe Pandas em Spark Dataframe
s_df.write.mode("overwrite").saveAsTable(table_name) # grava o DataFrame na Tabela Delta

#  ===========================----------



# ---------- dicas complementares ---------------------
# -----------------------------------------------
%sql
SELECT
  window.end AS dt_venda,
  SUM(qt_venda) AS total_itens_vendidos
FROM
  `academy`.`stock_name`.`vendas`
GROUP BY
  window(dt_venda, '90 days', '1 day')
  

# --------------------------------------------------
# Selecione o nome da empresa, stock,
#mínimo valor de fechamento, máximo valor de fechamento
#e percentual de variação entre o mínimo e o máximo valor de fechamento
#da tabela stock_bigtech
#agrupando por empresa e stock

%sql
SELECT
company,
stock,
MIN(close) AS min_close,
MAX(close) AS max_close,
100 * (MAX(close) - MIN(close)) / MIN(close) AS percent_variation
FROM academy.stock_name.stock_bigtech
GROUP BY company, stock


#---------------------------------------------

#Acrescente ao resultado a linha de concatenação com o nome da ação (STOCK),com o LINK (URL) de uma imagem.

%sql
SELECT
"https://raw.githubusercontent.com/Databricks-BR/genie_ai_bi/main/images/" || stock || ".png" AS image,
company,
stock,
MIN(close) AS min_close,
MAX(close) AS max_close,
100 * (MAX(close) - MIN(close)) / MIN(close) AS percent_variation
FROM academy.stock_name.stock_bigtech
GROUP BY company, stock

#Qual o faturamento em out/22?
%sql
SELECT
`stock_bigtech`.`stock`,
SUM(`stock_bigtech`.`close` * `stock_bigtech`.`volume`) AS `total_revenue_oct_2022`
FROM
`academy`.`stock_name`.`stock_bigtech`
WHERE
`stock_bigtech`.`date` >= '2022-10-01'
AND `stock_bigtech`.`date` <= '2022-10-31'
GROUP BY
`stock_bigtech`.`stock`

#Mantenha somente os 10 produtos com maior faturamento
%sql
SELECT
`stock_bigtech`.`stock`,
SUM(`stock_bigtech`.`close` * `stock_bigtech`.`volume`) AS `total_revenue_oct_2022`
FROM
`academy`.`stock_name`.`stock_bigtech`
WHERE
`stock_bigtech`.`date` >= '2022-10-01'
AND `stock_bigtech`.`date` <= '2022-10-31'
GROUP BY
`stock_bigtech`.`stock`
ORDER BY
`total_revenue_oct_2022` DESC
LIMIT 10


# new => add or upload data => create or modify table
# no databricks

#Qual o faturamento em out/22?
%sql
SELECT
SUM(`vendas`.`vl_venda`) AS faturamento_out_22
FROM
`academy`.`stock_name`.`vendas`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'

# faturamento
%sql
SELECT
`vendas`.`id_produto`,
SUM(`vendas`.`vl_venda`) AS faturamento_out_22
FROM
`academy`.`stock_name`.`vendas`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`vendas`.`id_produto`


# Mantenha somente os 10 produtos com maior faturamento

SELECT
`vendas`.`id_produto`,
SUM(`vendas`.`vl_venda`) AS faturamento_out_22
FROM
`academy`.`stock_name`.`vendas`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`vendas`.`id_produto`
ORDER BY
faturamento_out_22 DESC
LIMIT 10

# detalhe do nome dos produtos
%sql
SELECT
`dim_medicamento`.`nome_medicamento`,
SUM(`vendas`.`vl_venda`) AS faturamento_out_22
FROM
`academy`.`stock_name`.`vendas`
JOIN `academy`.`stock_name`.`dim_medicamento`
ON `vendas`.`id_produto` = `dim_medicamento`.`id_produto`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`dim_medicamento`.`nome_medicamento`
ORDER BY
faturamento_out_22 DESC
LIMIT 10



# detalhes
%sql
SELECT
`dim_medicamento`.`nome_medicamento`,
SUM(`vendas`.`vl_venda`) AS faturamento_out_22
FROM
`academy`.`stock_name`.`vendas`
JOIN `academy`.`stock_name`.`dim_medicamento`
ON `vendas`.`id_produto` = `dim_medicamento`.`id_produto`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`dim_medicamento`.`nome_medicamento`
ORDER BY
`dim_medicamento`.`nome_medicamento` ASC

#Qual o total de produtos vendidos em genéricos?
%sql
SELECT
SUM(`vendas`.`qt_venda`) AS total_genericos_vendidos
FROM
`academy`.`stock_name`.`vendas`
JOIN `academy`.`stock_name`.`dim_medicamento`
ON `vendas`.`id_produto` = `dim_medicamento`.`id_produto`
WHERE
`dim_medicamento`.`categoria_regulatoria` = 'GENÉRICO'

#Qual o valor total vendido de ansiolíticos?
%sql
SELECT
SUM(`vendas`.`vl_venda`) AS valor_total_ansio
FROM
`academy`.`stock_name`.`vendas`
JOIN `academy`.`stock_name`.`dim_medicamento`
ON `vendas`.`id_produto` = `dim_medicamento`.`id_produto`
WHERE
`dim_medicamento`.`classe_terapeutica` ILIKE '%ANSIOLITICO%'

#Quais produtos tiveram uma proporção de vendas por estoque maior que 0.8 em Outubro de 2022?

WITH vendas_agrupadas AS (
SELECT
`vendas`.`id_produto`,
SUM(`vendas`.`qt_venda`) AS total_vendido
FROM
`academy`.`stock_name`.`vendas`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`vendas`.`id_produto`
),
estoque_agrupado AS (
SELECT
`estoque`.`id_produto`,
SUM(`estoque`.`estoque`) AS estoque_total_mes
FROM
`academy`.`stock_name`.`estoque`
WHERE
`estoque`.`data_estoque` >= '2022-10-01'
AND `estoque`.`data_estoque` <= '2022-10-31'
GROUP BY
`estoque`.`id_produto`
)
SELECT
vendas_agrupadas.id_produto,
vendas_agrupadas.total_vendido,
estoque_agrupado.estoque_total_mes,
(
try_divide(vendas_agrupadas.total_vendido, estoque_agrupado.estoque_total_mes)
) AS proporcao_venda_estoque
FROM
vendas_agrupadas
JOIN estoque_agrupado
ON vendas_agrupadas.id_produto = estoque_agrupado.id_produto
WHERE
estoque_agrupado.estoque_total_mes > 0
AND (try_divide(vendas_agrupadas.total_vendido, estoque_agrupado.estoque_total_mes)) > 0.8





# versão nova
%sql
WITH vendas_agrupadas AS (
SELECT
`vendas`.`id_produto`,
SUM(`vendas`.`qt_venda`) AS total_vendido
FROM
`academy`.`stock_name`.`vendas`
WHERE
`vendas`.`dt_venda` >= '2022-10-01'
AND `vendas`.`dt_venda` <= '2022-10-31'
GROUP BY
`vendas`.`id_produto`
),
estoque_agrupado AS (
SELECT
`estoque`.`id_produto`,
SUM(`estoque`.`estoque`) AS estoque_total_mes
FROM
`academy`.`stock_name`.`estoque`
WHERE
`estoque`.`data_estoque` >= '2022-10-01'
AND `estoque`.`data_estoque` <= '2022-10-31'
GROUP BY
`estoque`.`id_produto`
)
SELECT
vendas_agrupadas.id_produto,
dim_medicamento.nome_medicamento,
vendas_agrupadas.total_vendido,
estoque_agrupado.estoque_total_mes,
(
try_divide(vendas_agrupadas.total_vendido, estoque_agrupado.estoque_total_mes)
) AS proporcao_venda_estoque
FROM
vendas_agrupadas
JOIN estoque_agrupado
ON vendas_agrupadas.id_produto = estoque_agrupado.id_produto
JOIN `academy`.`stock_name`.`dim_medicamento` AS dim_medicamento
ON vendas_agrupadas.id_produto = dim_medicamento.id_produto
WHERE
estoque_agrupado.estoque_total_mes > 0
AND (try_divide(vendas_agrupadas.total_vendido, estoque_agrupado.estoque_total_mes)) > 0.8


#Qual o valor total de venda por loja? Exiba o nome da loja
%sql
ALTER TABLE dim_loja ALTER COLUMN nlj COMMENT 'Nome da loja'


#Usando chaves primárias
#Aparentemente, a coluna id_loja da tabela dim_loja não é o melhor campo para fazer os cruzamentos com a tabela de vendas. Na verdade, a coluna correta é a cod!
#Vamos então adicionar chaves primárias e estrangeiras nessas tabelas para que a Genie não precise inferir como fazer esse cruzamento!

%sql
alter table dim_loja alter column cod set not null;
ALTER TABLE dim_loja ADD CONSTRAINT pk_dim_loja PRIMARY KEY (cod);
ALTER TABLE vendas ADD CONSTRAINT fk_venda_dim_loja FOREIGN KEY (id_loja) REFERENCES dim_loja(cod);

#Qual o valor total de venda por loja? Exiba o nome da loja
%sql
SELECT
`dim_loja`.`nlj` AS nome_loja,
SUM(`vendas`.`vl_venda`) AS valor_total_venda
FROM
`academy`.`stock_name`.`vendas`
JOIN `academy`.`stock_name`.`dim_loja`
ON `vendas`.`id_loja` = `dim_loja`.`cod`
GROUP BY
`dim_loja`.`nlj`


#* para calcular indicadores sobre prescrição use categoria_regulatoria <> 'GENÉRICO'

#Calcule a quantidade de itens vendidos por janela móvel de 3 meses
%sql
SELECT
`vendas`.`dt_venda`,
SUM(`vendas`.`qt_venda`) AS qt_venda_dia,
SUM(SUM(`vendas`.`qt_venda`)) OVER (
ORDER BY `vendas`.`dt_venda`
RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW
) AS qt_venda_3m
FROM
`academy`.`stock_name`.`vendas`
GROUP BY
`vendas`.`dt_venda`
ORDER BY
`vendas`.`dt_venda`

# versão adaptada e correta da janela móvel de 3 meses
%sql
SELECT
window.end AS dt_venda,
SUM(qt_venda) AS total_itens_vendidos
FROM
`academy`.`stock_name`.`vendas`
GROUP BY
window(dt_venda, '90 days', '1 day')

#Qual o lucro projetado do AAS?
%sql
CREATE OR REPLACE FUNCTION calc_lucro(medicamento STRING)
RETURNS TABLE(nome_medicamento STRING, lucro_projetado DOUBLE)
COMMENT 'Use esta função para calcular o lucro projetado de um medicamento'
RETURN 
SELECT
m.nome_medicamento,
sum(case when m.categoria_regulatoria == 'GENÉRICO' then 1 else 0.5 end * v.vl_venda) / sum(v.qt_venda) as lucro_projetado
FROM vendas v
LEFT JOIN dim_medicamento m
ON v.id_produto = m.id_produto
WHERE m.nome_medicamento = calc_lucro.medicamento
GROUP BY ALL



