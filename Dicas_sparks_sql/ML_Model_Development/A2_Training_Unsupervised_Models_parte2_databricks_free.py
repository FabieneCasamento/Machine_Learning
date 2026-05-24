# Databricks notebook source
try:
    cluster_name = spark.conf.get("spark.databricks.clusterUsageTags.clusterName")
except Exception:
    cluster_name = "Cluster name not available (Serverless compute)"
print(f"Nome do cluster atual: {cluster_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 1.1b Demo - Unsupervised Learning
# MAGIC
# MAGIC ## Training Unsupervised Models
# MAGIC
# MAGIC In this demo, we will explore **unsupervised learning**, a method where the model finds patterns in **unlabeled data** without predefined categories. We will use **text embeddings** to convert text into numerical representations and apply **K-Means clustering** to group similar text documents.
# MAGIC
# MAGIC To improve clustering efficiency, we will **reduce the dimensionality** of embeddings using **Principal Component Analysis (PCA)**. We will also use evaluation techniques like the **Elbow Method** and **Silhouette Score** to determine the best number of clusters and assess clustering quality.
# MAGIC
# MAGIC ### Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you will be able to:
# MAGIC
# MAGIC * **Generate text embeddings** using the embeddings model *General Text Embeddings (GTE)* to represent text numerically.
# MAGIC * **Apply dimensionality reduction (PCA)** to optimize clustering performance.
# MAGIC * **Train an unsupervised K-Means model** to discover patterns in text data.
# MAGIC * **Determine the optimal number of clusters** using the *Elbow Method*.
# MAGIC * **Evaluate clustering quality** using *Silhouette Score*.
# MAGIC * **Visualize clustering results** for better interpretability.

# COMMAND ----------

# Instalar e atualizar as dependências necessárias para agrupamento e avaliação
%pip install --upgrade threadpoolctl scikit-learn
%pip install kneed

# Reiniciar o kernel do Python para aplicar as atualizações de pacotes imediatamente
%restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements and Environment Setup
# MAGIC
# MAGIC Before starting the unsupervised learning pipeline, we need to ensure our environment has the latest version of `scikit-learn` and install `kneed` to help us mathematically determine the optimal number of clusters during the Elbow Method evaluation.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following environment requirements before starting the unsupervised learning lesson:
# MAGIC
# MAGIC * **Databricks Runtime (Original):** To run this notebook seamlessly in its native cloud environment, a cluster with Databricks Runtime **16.0.x-cpu-ml-scala2.12** (or higher) containing pre-installed Machine Learning libraries is expected.
# MAGIC * **Local Alternative:** If executing locally on a Jupyter environment, make sure your Python installation handles thread orchestration correctly by having updated packages for `scikit-learn` and `threadpoolctl`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## REQUIRED - SELECT CLASSIC COMPUTE
# MAGIC
# MAGIC Before executing cells in this notebook, it is important to understand the compute context. In a standard cloud lab environment, **Serverless** compute may be enabled by default. However, complex multi-threaded clustering pipelines often perform better or require explicit allocation on a classic single-node or multi-node compute cluster.
# MAGIC
# MAGIC ### Steps to configure or verify compute nodes (Cloud Architecture Context):
# MAGIC
# MAGIC 1. **Navigate to the top-right** of this notebook interface and click the compute drop-down menu to review your cluster allocation.
# MAGIC 2. **If Serverless is active** and you experience thread pool blockages during PCA or K-Means optimization, switch to a designated **Classic Compute Resource** under the *More* options.
# MAGIC 3. **If your cluster has terminated**, remember to restart it from the *Compute* tab on the left navigation pane before triggering heavy mathematical operations such as multi-k inertia scoring.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before shifting focus directly onto text embeddings, we must invoke the baseline configuration scripts. In an enterprise workspace setup, these utility helper lines define localized folder variables, clear old cached parquet outputs, and prepare tracking variables for MLflow logging tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data & Generate Embeddings
# MAGIC
# MAGIC Before we can apply unsupervised learning, we need to load and process our dataset. In this step, we will:
# MAGIC
# MAGIC * **Load the AG News dataset** from our data source.
# MAGIC * **Extract the `text` column** for downstream natural language processing.
# MAGIC * **Prepare the data** for embedding generation.
# MAGIC
# MAGIC ### Load the Dataset
# MAGIC
# MAGIC We use the **AG News dataset**, which contains news articles, to perform text clustering. The text features within this dataset will allow our model to find semantic patterns and group similar articles automatically.

# COMMAND ----------

# MAGIC %md
# MAGIC https://www.kaggle.com/code/sujithmandala/simple-feature-extractor-bert-model/notebook

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import col

# COMMAND ----------

# DBTITLE 1,Install OpenAI dependencies
# MAGIC %pip install databricks-sdk[openai]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType


# COMMAND ----------

# Install the Kaggle API client
%pip install kaggle

# Make sure the .kaggle directory exists
!mkdir -p ~/.kaggle

# Move kaggle.json to the correct directory and set permissions
# This assumes you have uploaded kaggle.json to the current /content/ directory
# If you uploaded it to a different path, adjust accordingly.
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

print("Kaggle API key setup complete.")

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, concat, lit

# 1. Carregar usando Pandas direto de uma URL pública do AG News (garante que funciona em qualquer lugar)
url_ag_news = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"

print("A carregar amostra via Pandas...")
# Lemos apenas as 1000 linhas desejadas direto para a memória
pandas_df = pd.read_csv(url_ag_news, header=None, nrows=1000, names=["Class Index", "Title", "Description"])

# 2. Definir o esquema do Spark correspondente
schema = StructType([
    StructField("Class Index", IntegerType(), True),
    StructField("Title", StringType(), True),
    StructField("Description", StringType(), True)
])

print("A converter para DataFrame do PySpark...")
# 3. Transformar o DataFrame do Pandas em um DataFrame do PySpark distribuído
news_df = spark.createDataFrame(pandas_df, schema=schema)

# 4. Aplicar a lógica de unificação de texto ('Title' + 'Description') conforme a aula
news_texts_df = news_df.select(
    concat(col("Title"), lit(" - "), col("Description")).alias("text")
)

# 5. Exibir o resultado final pronto para os Embeddings
print(f"Sucesso! Total de linhas no Spark: {news_texts_df.count()}")
display(news_texts_df)

# COMMAND ----------

# Define the Kaggle dataset slug
#kaggle_dataset_slug = "amananandj/ag-news-dataset" # This is a common public AG News dataset
# The path from the original cell implies 'new-dataset-for-text-classification-ag-news'.
# If the previous slug causes an error, try this one, but be aware it might be a user's private dataset.
# kaggle_dataset_slug = "new-dataset-for-text-classification-ag-news"

# Download the dataset
#!kaggle datasets download -d {kaggle_dataset_slug}

# Unzip the dataset
# This assumes the downloaded zip file is named after the dataset slug
#!unzip -o {kaggle_dataset_slug.split('/')[-1]}.zip -d {kaggle_dataset_slug.split('/')[-1]}

#print(f"Dataset downloaded and extracted to ./{kaggle_dataset_slug.split('/')[-1]}/")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download AG News Dataset from Kaggle
# MAGIC
# MAGIC To download the dataset from Kaggle, you'll need a Kaggle API key.
# MAGIC
# MAGIC 1.  Go to [Kaggle](https://www.kaggle.com/).
# MAGIC 2.  Click on your profile picture (top right) and select "Account".
# MAGIC 3.  Scroll down to the "API" section and click "Create New API Token". This will download `kaggle.json`.
# MAGIC 4.  Upload this `kaggle.json` file to your Google Colab environment. In the left sidebar, click the "Files" icon, then upload the file to your root directory (e.g., `/content/`).

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we will download the specific AG News dataset from Kaggle. The dataset ID is derived from the path you provided.

# COMMAND ----------

# MAGIC %md
# MAGIC The `train.csv` file should now be available locally within the extracted dataset folder. We will now modify the original cell to load it from this local path.

# COMMAND ----------

display(news_texts_df.show(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Text Embeddings Using gte-large
# MAGIC
# MAGIC Now that we have loaded our text dataset, the next step is to convert the text data into **numerical representations** using **text embeddings**. Here we will demonstrate how easy it is to take our text and embed it using a foundational model from Mosaic AI Model Serving.
# MAGIC
# MAGIC In particular, we will use the `get_open_ai_client()` method, which is part of the Databricks SDK, providing a convenient way to create an OpenAI-compatible client for interacting with the foundation model.
# MAGIC
# MAGIC ### Execution Steps:
# MAGIC
# MAGIC * **Step 1:** Initialize OpenAI Client via Databricks Workspace SDK.
# MAGIC * **Step 2:** Define Embedding Function targeting `databricks-gte-large-en`.
# MAGIC * **Step 3:** Convert Text to Dense Embeddings.
# MAGIC * **Step 4:** Convert embeddings list back to a Spark DataFrame.

# COMMAND ----------

# DBTITLE 1,Cell 19
import time
from pyspark.sql.functions import col
from databricks.sdk import WorkspaceClient

# 1. Inicializar o cliente OpenAI nativo via Databricks Workspace SDK
workspace_client = WorkspaceClient()
openai_client = workspace_client.serving_endpoints.get_open_ai_client()

# 2. Definir a função que envia um lote de textos para o endpoint gte-large com retry
def get_embeddings_batch(text_batch, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = openai_client.embeddings.create(
                model="databricks-gte-large-en",
                input=text_batch
            )
            return [res.embedding for res in response.data]
        except Exception as e:
            if "429" in str(e) or "RATE_LIMIT" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 10  # Exponential backoff: 10s, 20s, 40s, 80s, 160s
                    print(f"Rate limit atingido. Aguardando {wait_time}s antes de tentar novamente...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise

# 3. Coletar o Spark DataFrame de textos para uma lista Python no Driver
print("Coletando textos para o driver...")
news_texts_list = news_texts_df.select("text").toPandas()["text"].tolist()

# Reduzir dataset para 50 registros para respeitar limites de taxa do workspace
print(f"Reduzindo dataset de {len(news_texts_list)} para 50 registros para respeitar rate limits...")
news_texts_list = news_texts_list[:50]

# 4. Processar os dados em lotes (Batches) para respeitar limites de taxa (Rate Limits)
batch_size = 10  # Reduzir batch size para 10
embeddings_list = []

print(f"Gerando embeddings para {len(news_texts_list)} registros em lotes de {batch_size}...")
print("Aguardando 30 segundos antes de iniciar para permitir que rate limits se recuperem...")
time.sleep(30)

for i in range(0, len(news_texts_list), batch_size):
    batch = news_texts_list[i : i + batch_size]
    print(f"Processando lote {i//batch_size + 1}/{(len(news_texts_list)-1)//batch_size + 1}...")
    embeddings_list.extend(get_embeddings_batch(batch))
    
    # Adicionar delay entre lotes para respeitar o limite de taxa do workspace
    if i + batch_size < len(news_texts_list):
        time.sleep(10)  # Aumentar delay para 10 segundos

# 5. Criar o Spark DataFrame final zipando a lista de textos e a lista de vetores
embeddings_df = spark.createDataFrame(
    zip(news_texts_list, embeddings_list),
    ["text", "embedding"]
)

# Validar o esquema final gerado
print("Estrutura do DataFrame de Embeddings:")
embeddings_df.printSchema()
print(f"Total de embeddings gerados: {embeddings_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC https://docs.databricks.com/aws/en/machine-learning/model-serving/query-embedding-models

# COMMAND ----------

embeddings_df.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC # base IA

# COMMAND ----------

# DBTITLE 1,Cell 22
from databricks.sdk import WorkspaceClient

client = WorkspaceClient()
openai_client = client.serving_endpoints.get_open_ai_client()

response = openai_client.embeddings.create(
  model="databricks-gte-large-en",
  input="what is databricks"
)

# COMMAND ----------

response

# COMMAND ----------

#Verificar o tipo e a estrutura do objeto response
print(f"Tipo: {type(response)}")
print(f"\nAtributos disponíveis:")
print([attr for attr in dir(response) if not attr.startswith('_')])

# Tentar acessar os dados corretamente
print(f"\nresponse.data: {response.data}")
print(f"\nPrimeiro embedding (primeiros 10 valores):")
print(response.data[0].embedding[:10])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardization and Dimensionality Reduction
# MAGIC
# MAGIC Now that we have generated text embeddings, we need to prepare them for clustering by applying **standardization** and **dimensionality reduction**.
# MAGIC
# MAGIC ### Why Do We Need This Step?
# MAGIC
# MAGIC * **Standardization:** Ensures that all features have a similar scale, preventing some features from dominating others during distance calculations.
# MAGIC * **Dimensionality Reduction:** Using **Principal Component Analysis (PCA)** helps reduce the number of features while retaining critical semantic information. This makes clustering more computationally efficient and easier to visualize. In particular, we will be converting our embeddings from **1024 dimensions down to 2 dimensions**.

# COMMAND ----------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# 1. Converter o Spark DataFrame para um Array NumPy para processamento no Scikit-Learn
print("A extrair os vetores de embedding para processamento matricial...")
embeddings_np = np.array([row["embedding"] for row in embeddings_df.select("embedding").collect()])

# Step 1: Standardization
print("A aplicar o StandardScaler...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_np)

# Step 2: Dimensionality Reduction using PCA (Redução para 2 componentes)
print("A executar a redução de dimensionalidade (PCA -> 2D)...")
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings_scaled)

# 2. Converter de volta para um Spark DataFrame contendo as coordenadas das componentes principais
print("A reconverter os dados transformados para o PySpark...")
pca_data_tuples = [(int(i), float(pc[0]), float(pc[1])) for i, pc in enumerate(embeddings_pca)]

# Mapeia as colunas exatamente como exigido no laboratório: unique_id, PC1, PC2
pca_df = spark.createDataFrame(pca_data_tuples, ["unique_id", "PC1", "PC2"])

# 3. Exibir a tabela final resultante para validação
print("\nSucesso! Estrutura final das componentes prontas para o K-Means:")
display(pca_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine the Optimal Number of Clusters (Elbow Method)
# MAGIC
# MAGIC Before applying K-Means clustering, we need to determine the **best number of clusters ($K$)**. We use the **Elbow Method**, which helps identify the point where adding more clusters no longer significantly reduces inertia (the sum of squared distances of samples to their closest cluster center).
# MAGIC
# MAGIC ### How Does the Elbow Method Work?
# MAGIC * We run K-Means clustering for different values of $K$ (from 1 to 10).
# MAGIC * We measure **inertia** (how well points fit within their assigned cluster).
# MAGIC * We plot inertia against $K$ and look for the **elbow point**, where the decrease in inertia slows down.
# MAGIC * The **optimal $K$** is found programmatically using `KneeLocator`, which detects this elbow point automatically.
# MAGIC
# MAGIC ### Why not just minimize inertia?
# MAGIC * Minimizing inertia indefinitely can lead to **overfitting** (continuously decreasing while increasing the number of clusters will eventually fit noise rather than meaningful patterns).
# MAGIC * The elbow method provides interpretability and avoids arbitrary decision-making by providing a point of diminishing returns.
# MAGIC
# MAGIC > **Note:** We manually set the environment variable `OMP_NUM_THREADS` to `1` to avoid multi-threading conflicts and resource overhead, ensuring consistent performance during the optimization loop.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Determine the Optimal Number of Clusters (Elbow Method)
# MAGIC
# MAGIC Before applying K-Means clustering, we need to determine the **best number of clusters ($K$)**. We use the **Elbow Method**, which helps identify the point where adding more clusters no longer significantly reduces inertia (sum of squared distances to cluster centers).
# MAGIC
# MAGIC ### How Does the Elbow Method Work?
# MAGIC * We run K-Means clustering for different values of $K$ (from 1 to 10).
# MAGIC * We measure **inertia** (how well points fit within their assigned cluster).
# MAGIC * We plot inertia against $K$ and look for the **elbow point** where the decrease in inertia slows down.
# MAGIC * The **optimal $K$** is found programmatically using `KneeLocator`, which detects the elbow point automatically.
# MAGIC
# MAGIC ### Why not just minimize inertia?
# MAGIC * Minimizing inertia can lead to *overfitting* (continuously decreasing while increasing the number of clusters will fit noise rather than meaningful patterns).
# MAGIC * The elbow method provides interpretability and voids arbitrary decision-making by providing a point of diminishing returns.
# MAGIC
# MAGIC We manually set the environment variable `OMP_NUM_THREADS` to 1 to avoid multithreading and parallelism to ensure that each run uses the same computational resources. This prevents the creation of too many threads across processes, preventing inefficient CPU utilization.

# COMMAND ----------

import os
import threadpoolctl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

# Apply fixes for parallel processing (Garante consistência de threads no cluster)
os.environ["OMP_NUM_THREADS"] = "1"
threadpoolctl.threadpool_limits(limits=1, user_api="blas")

# Perform K-Means clustering and compute inertia
inertia = []
k_values = range(1, 10)  # Try values from 1 to 10

print("Running K-Means optimizations over embedding vector space...")
for k in k_values:
    # n_init=10 executa 10 inicializações independentes para evitar mínimos locais ruins
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings_scaled)  # Ensure embeddings_scaled is preprocessed
    inertia.append(kmeans.inertia_)
    print(f"  └── Clusters (K) = {k} | Inertia Score: {kmeans.inertia_:.2f}")

# Use KneeLocator to find the elbow point
# Scurve indica comportamento convexo decrescente padrão de curvas de inércia
knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
optimal_k = knee_locator.elbow

# Se KneeLocator não encontrou um elbow claro, usar heurística alternativa
if optimal_k is None:
    # Calcular a segunda derivada (variação na taxa de decréscimo)
    first_diff = np.diff(inertia)
    second_diff = np.diff(first_diff)
    # Encontrar o ponto onde a segunda derivada é máxima (maior mudança na taxa)
    # Adicionar 2 ao índice porque perdemos 2 pontos com as diferenças
    optimal_k = int(np.argmax(second_diff) + 2)
    print(f"\n[INFO] KneeLocator não detectou elbow claro. Usando heurística alternativa.")

print("\n" + "="*60)
print(f" MATHEMATICAL ANALYSIS COMPLETE: Optimal K discovered = {optimal_k}")
print("="*60 + "\n")

# Plot Elbow Method with detected optimal k
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o', linestyle='--', label='Inertia', color='#1f77b4')
plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K = {optimal_k}')

plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K', fontsize=12, fontweight='bold', pad=10)
plt.legend()
plt.grid(True, alpha=0.3)

# Salva o gráfico limpo no seu repositório local
plt.savefig('optimal_k_elbow_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# COMMAND ----------

optimal_k

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Clustering Algorithm
# MAGIC
# MAGIC We will now apply **K-Means Clustering** to group similar news articles together based on their embeddings.
# MAGIC
# MAGIC ### Steps:
# MAGIC 1. **Train the K-Means model** using the `optimal_k`.
# MAGIC 2. **Assign cluster labels** to each news article.
# MAGIC 3. **Store clustering results** in a Spark DataFrame.

# COMMAND ----------

# era para ser 4 o valor do cluster
#optimal_k =4

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id
from sklearn.cluster import KMeans
import pandas as pd

# # Apply K-Means clustering on the reduced embeddings
# Usamos o 'optimal_k' (ponto de cotovelo) identificado pelo KneeLocator
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(embeddings_scaled) # Fit the model on the standardized embeddings

# # Get cluster labels
labels = kmeans.labels_

# # Convert labels to a Spark DataFrame
# Mapeia cada label gerada ao seu respectivo identificador numérico sequencial
labels_df = pd.DataFrame({"unique_id": range(len(labels)), "Cluster": labels})
labels_spark_df = spark.createDataFrame(labels_df)

# # Join PCA-transformed Spark DataFrame with cluster labels
# Consolida as colunas geométricas (PC1, PC2) com a respectiva label do cluster
clusters_spark_df = pca_df.join(labels_spark_df, "unique_id")

# # Display the resulting clustered DataFrame
display(clusters_spark_df)

# COMMAND ----------

embeddings_scaled

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Clustering Performance
# MAGIC
# MAGIC Once the **K-Means clustering** is applied, we need to assess how well the clusters are formed. A common metric for this evaluation is the **Silhouette Score**.
# MAGIC
# MAGIC ### Silhouette Score
# MAGIC
# MAGIC *The silhouette value measures how well an object fits its assigned cluster compared to other clusters*, ranging from -1 to +1, with higher values indicating better clustering. It provides a metric for evaluating clustering quality, with average scores above 0.5 considered reasonable, though high-dimensional data may yield lower scores due to the curse of dimensionality.

# COMMAND ----------

from sklearn.metrics import silhouette_score

# Calculate the silhouette score to evaluate clustering quality
# O cálculo utiliza a matriz padronizada original e as labels geradas pelo K-Means
silhouette_avg = silhouette_score(embeddings_scaled, labels)

print(f"Silhouette Score for K-Means with {optimal_k} clusters: {silhouette_avg}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate Clustering Performance
# MAGIC
# MAGIC Once the **K-Means clustering** is applied, we need to assess how well the clusters are formed. A common metric for this evaluation is the **Silhouette Score**.
# MAGIC
# MAGIC ### Silhouette Score
# MAGIC
# MAGIC *The silhouette value measures how well an object fits its assigned cluster compared to other clusters*, ranging from -1 to +1, with higher values indicating better clustering. It provides a metric for evaluating clustering quality, with average scores above 0.5 considered reasonable, though high-dimensional data may yield lower scores due to the curse of dimensionality.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Clustering Results
# MAGIC
# MAGIC We will visualize the clusters to gain insights into how the news articles are grouped based on their embeddings. Here we will be using the method `ConvexHull` to help visualize. This computes the convex hull in $N$ dimensions (here $N = 2$). This helps us identify the boundary of a set of clusters.

# COMMAND ----------



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull

# # Convert Spark DataFrame to Pandas
clusters_pd = clusters_spark_df.toPandas()

# # Define color palette
num_clusters = clusters_pd["Cluster"].nunique()
colors = sns.color_palette("husl", num_clusters) # Distinct colors

plt.figure(figsize=(10, 7))

# # Scatter plot with better visibility
for cluster, color in zip(range(num_clusters), colors):
    subset = clusters_pd[clusters_pd["Cluster"] == cluster]
    
    plt.scatter(
        subset["PC1"], subset["PC2"],
        label=f"Cluster {cluster}",
        color=color, s=80, alpha=0.6, edgecolors='k' # Larger points, transparency, black edges
    )
    
    # # Convex Hull for cluster boundary (only if there are enough points)
    if len(subset) > 2:
        hull = ConvexHull(subset[["PC1", "PC2"]])
        for simplex in hull.simplices:
            plt.plot(subset.iloc[simplex]["PC1"], subset.iloc[simplex]["PC2"], color=color, alpha=0.5)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Clustering Visualization of News Articles")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# COMMAND ----------

