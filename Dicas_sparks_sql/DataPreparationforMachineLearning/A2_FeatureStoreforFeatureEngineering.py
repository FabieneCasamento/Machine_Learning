# Databricks notebook source
#%run ../Includes/Classrroom-Setup-3.1

# COMMAND ----------

# MAGIC %sh pwd

# COMMAND ----------

# MAGIC %sh find /Workspace -name "00-setup*"

# COMMAND ----------

# DBTITLE 1,Cell 4
# Tente este comando padrão:
# Criando um objeto DA manual para substituir o que está faltando
class FakeDA:
    def __init__(self):
        # Pega seu usuário logado automaticamente
        self.username = spark.sql("SELECT current_user()").collect()[0][0]
        # Define nomes padrão para catálogo e esquema
        self.catalog_name = "workspace" 
        self.schema_name = "default"
        
        class Paths:
            def __init__(self):
                self.working_dir = f"/Volumes/workspace/default/working_dir"
                # Local padrão de datasets públicos do Databricks
                self.datasets = "/databricks-datasets" 
        
        self.paths = Paths()

# Inicializa o objeto
DA = FakeDA()

# Agora o seu comando de print vai funcionar:
print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

try:
    # Definindo as variáveis de caminho baseadas no print
    shared_volume_name = 'telco' # From Marketplace
    csv_name = 'telco-customer-churn' # CSV file name

    # Caminho completo (ajustado para o padrão do curso)
    dataset_path = f"{DA.paths.datasets.telco}/{shared_volume_name}/{csv_name}.csv"
    #dataset_path= "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    # Carregando o dataset com Spark
    telco_df = spark.read.csv(dataset_path, header="true", inferSchema="true", multiLine="true", escape='"')

    # Removendo a coluna alvo (Target) conforme instrução da aula
    telco_df = telco_df.drop("Churn")

    # Visualizando os dados
    display(telco_df)

    # Geralmente a aula anterior salva os dados limpos aqui:
    telco_df = spark.read.table(f"{DA.catalog_name}.{DA.schema_name}.cleaned_telco")
    print("Dados carregados da tabela limpa (aula anterior)!")
except:
    print("Tabela não encontrada. Use o comando do CSV acima.")

# COMMAND ----------

import pandas as pd

print("Tabela não encontrada ou objeto DA não definido. Carregando direto do GitHub...")
    
    # 2. Caminho do GitHub
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
# 3. O Spark não lê HTTPS direto, então usamos Pandas como ponte
pdf = pd.read_csv(url)
    
# 4. Converte o DataFrame do Pandas para Spark DataFrame
telco_df = spark.createDataFrame(pdf)
    
# 5. Removendo a coluna alvo (Target) conforme instrução da aula
telco_df = telco_df.drop("Churn")
    
print("Dados carregados com sucesso via GitHub/Pandas!")

# Visualizando os dados
display(telco_df)

# COMMAND ----------

print(telco_df.show(5))


# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# 1. INICIALIZAÇÃO
# O FeatureEngineeringClient é a ponte entre o seu código e o Unity Catalog.
# Ele gerencia a linhagem (lineage) e o versionamento das suas colunas.
fe = FeatureEngineeringClient()


# COMMAND ----------

#import fe
help(fe.create_table)


# COMMAND ----------

spark.sql("SHOW CATALOGS").display()

#catalog
#samples
#system
#workspace

# COMMAND ----------

spark.sql("SELECT current_catalog(), current_schema()").display()
#current_catalog()	current_schema()
#workspace	default

# COMMAND ----------

# DBTITLE 1,Cell 12

# 2. DEFINIÇÃO DE METADADOS
# No Databricks, tabelas de features devem seguir o formato: catalogo.esquema.tabela
# Se o objeto DA não funcionar, substitua por strings manuais como 'workspace.default.telco_features'
table_name = f"{DA.catalog_name}.{DA.schema_name}.telco_features"

# 3. CRIAÇÃO DA TABELA DE RECURSOS (Feature Table)
# O método create_table registra o DataFrame no Feature Store.
fe.create_table(
    name=table_name,            # Nome único no catálogo para governança.
    primary_keys=["customerID"], # O ID único que permite ao modelo buscar o dado em tempo real.
    df=telco_df,                # O DataFrame carregado anteriormente (sem a coluna alvo Churn).
    description="Features de comportamento do cliente extraídas do dataset Telco.",
    tags= {"source":"bronze","format":"delta"}
)

"""
EXPLICAÇÃO TÉCNICA DO SCRIPT:
----------------------------
1. Por que usamos 'primary_keys'? 
   Diferente de tabelas comuns, o Feature Store exige uma chave. Isso permite que, 
   na fase de inferência (previsão), o modelo receba apenas o ID do cliente e 
   busque automaticamente todas as características (idade, contrato, etc.) nesta tabela.

2. Por que registrar como 'Feature Table' e não uma 'Delta Table' comum?
   - Automatização: O Databricks rastreia quais colunas foram usadas em qual modelo.
   - Reutilização: Outros cientistas podem usar as mesmas colunas sem recalcular tudo.
   - Online Serving: Esta tabela pode ser sincronizada com bancos de baixa latência (como CosmosDB ou DynamoDB) 
     para previsões em milissegundos em sites/apps.
"""
#workspace.default.telco_features => no catalog
print(f"Sucesso! A tabela {table_name} agora está disponível no Feature Store.")

# COMMAND ----------

# Busca o objeto da tabela de recursos usando o nome definido anteriormente
ft = fe.get_table(name=table_name)

# Exibe a descrição que você definiu na criação
print(f"Feature Table description: {ft.description}")

# Exibe a lista de todas as colunas (features) registradas na tabela
print(ft.features)
#Feature Table description: Features de comportamento do cliente extraídas do dataset Telco.
#['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# COMMAND ----------

# Carrega os dados da Feature Table para um DataFrame e exibe na tela
display(fe.read_table(name=table_name))

# COMMAND ----------

from pyspark.sql.functions import when

# Criar uma nova coluna 'tenure_group' baseada na coluna 'tenure'
# Categoriza a permanência do cliente em: short, mid e long
telco_df_updated = telco_df.withColumn("tenure_group", 
    when((telco_df.tenure >= 0) & (telco_df.tenure <= 25), "short")
    .when((telco_df.tenure > 25) & (telco_df.tenure <= 50), "mid")
    .when((telco_df.tenure > 50) & (telco_df.tenure <= 75), "long")
    .otherwise("invalid")
)

# COMMAND ----------

# Escrever a nova feature de volta para a Feature Table existente
fe.write_table(
    name=table_name,
    # Selecionamos a Primary Key e a nova coluna apenas
    df=telco_df_updated.select("customerID", "tenure_group"),
    mode="merge" # Parâmetro crucial para adicionar sem apagar o que já existe
)

# COMMAND ----------

# MAGIC %md
# MAGIC foco é a manutenção e versionamento das tabelas de recursos. O script mostra como deletar uma coluna e como usar o "Time Travel" do Delta Lake para recuperar dados de uma versão anterior.
# MAGIC

# COMMAND ----------

# --- CÉLULA 26: DELETAR UMA FEATURE EXISTENTE ---
# Para deletar colunas em tabelas Delta (usadas pelo Feature Store), 
# precisamos habilitar o mapeamento de colunas (Column Mapping).
spark.sql(f"""
ALTER TABLE {table_name} SET TBLPROPERTIES (
    'delta.columnMapping.mode' = 'name',
    'delta.minReaderVersion' = '2',
    'delta.minWriterVersion' = '5'
)
""")

# COMMAND ----------

# Agora podemos remover a coluna 'tenure' original, já que criamos o 'tenure_group'
spark.sql(f"ALTER TABLE {table_name} DROP COLUMNS (tenure)")

# COMMAND ----------

# --- CÉLULA 28: BUSCAR O HISTÓRICO (VERSIONAMENTO) ---
# Tabelas de features herdam o versionamento do Delta Lake.
# Aqui buscamos o timestamp de uma versão específica (neste caso, a versão 3)
timestamp_v3 = spark.sql(f"DESCRIBE HISTORY {table_name}") \
    .orderBy("version") \
    .collect()[3].timestamp

print(f"Timestamp da Versão 3: {timestamp_v3}")

# COMMAND ----------

timeteste = spark.sql(f"DESCRIBE HISTORY {table_name}") \
    .orderBy("version") \
    .collect()[0].timestamp

print(f"Timestamp da Versão 0: {timeteste}")

# COMMAND ----------

# --- CÉLULA 29: TIME TRAVEL (VIAGEM NO TEMPO) ---
# Como ler a tabela exatamente como ela era no passado usando o Spark nativo
telco_df_v3 = (spark
    .read
    .option("timestampAsOf", timestamp_v3) # Define o ponto exato no tempo
    .table(table_name))

display(telco_df_v3)

# COMMAND ----------

timev2= spark.sql(f"DESCRIBE HISTORY {table_name}") \
    .orderBy("version") \
    .collect()[2].timestamp

print(f"Timestamp da Versão 2: {timev2}")

# --- CÉLULA 29: TIME TRAVEL (VIAGEM NO TEMPO) ---
# Como ler a tabela exatamente como ela era no passado usando o Spark nativo
telco_df_v2 = (spark
    .read
    .option("timestampAsOf", timev2) # Define o ponto exato no tempo
    .table(table_name))

display(telco_df_v2)

# COMMAND ----------

SHOW TABLES IN workspace.default


# COMMAND ----------

# DBTITLE 1,Cell 26
# --- CÉLULA 32: CONVERTENDO UMA TABELA DELTA EM FEATURE TABLE ---

# Primeiro, visualizamos a tabela existente que contém recursos de telco
# Esta é uma tabela Delta comum que reside no seu esquema do Unity Catalog.
display(spark.sql("SELECT * FROM workspace.default.telco_features"))

# COMMAND ----------

# DBTITLE 1,Cell 26
# --- CÉLULA 32: CONVERTENDO UMA TABELA DELTA EM FEATURE TABLE ---

# Primeiro, visualizamos a tabela existente que contém recursos de telco
# Esta é uma tabela Delta comum que reside no seu esquema do Unity Catalog.
display(spark.sql("SELECT * FROM workspace.default.telco_features"))

# COMMAND ----------

# A. Definir a coluna da Chave Primária (customerID) como NOT NULL
#spark.sql(f"ALTER TABLE {DA.catalog_name}.{DA.schema_name}.security_features "
#          "ALTER COLUMN customerID SET NOT NULL")

# B. Adicionar a restrição de Chave Primária (Primary Key Constraint)
#spark.sql(f"ALTER TABLE {DA.catalog_name}.{DA.schema_name}.security_features "
#          "ADD CONSTRAINT security_pk PRIMARY KEY(customerID)")

# COMMAND ----------

