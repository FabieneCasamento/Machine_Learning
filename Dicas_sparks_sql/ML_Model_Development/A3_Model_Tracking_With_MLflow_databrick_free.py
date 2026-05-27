# Databricks notebook source
# MAGIC %md
# MAGIC # Model Tracking with MLflow
# MAGIC
# MAGIC In this demo, we will explore the capabilities of MLflow, a comprehensive framework for the complete machine learning lifecycle. MLflow provides tools for tracking experiments, packaging code into reproducible runs, and sharing and deploying models.
# MAGIC
# MAGIC
# MAGIC In this demo, we will focus on tracking and logging components of MLflow. First, we will demonstrate how to track an experiment with MLflow and show various custom logging features including loggin parameters, metrics, figures and arbitrary artifacts.
# MAGIC
# MAGIC ### Learning Objectives:
# MAGIC
# MAGIC By the end of this demo, you will be able to;
# MAGIC
# MAGIC Train a model using a Feature Store table as the modeling set.
# MAGIC
# MAGIC Manually log parameters, metrics, models, and figures with MLflow tracking.
# MAGIC
# MAGIC Log training dataset with model in MLflow
# MAGIC
# MAGIC Log additional artifacts to a model run
# MAGIC
# MAGIC Review an experiment using the MLflow UI.

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow with Unity Catalog
# MAGIC
# MAGIC Databricks has support for MLflow with Unity Catalog (UC) integration and workspace based classic version. Although we won't go into the details of MLflow with UC in this demo, we will enable it. This means models will be registered to UC.

# COMMAND ----------

# MAGIC %pip install --upgrade 'mlflow-skinny[databricks]'
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Dataset
# MAGIC
# MAGIC ## Load Dataset
# MAGIC
# MAGIC In this section, we will leverage the Feature Store to load the dataset for our machine learning experiment. Instead of directly reading from a CSV file, we will use the Feature Store setup to create a feature table and then read the data from it. This approach enhances reproducibility and ensures consistency in the datasets used for training and testing.

# COMMAND ----------

#!pip install mlflow

# COMMAND ----------

import mlflow
import pandas as pd

# URL do arquivo bruto (raw) do GitHub com o Diabetes dataset
url_diabetes = "https://raw.githubusercontent.com/GRicciardi00/Kaggle-Diabetes-classification/main/diabete.csv"

print("A carregar o dataset de Diabetes a partir do GitHub...")
feature_data_pd = pd.read_csv(url_diabetes)

# COMMAND ----------

feature_data_pd.columns

# COMMAND ----------


#feature_dataset = mlflow.data.load_delta(
#    table_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes_binary",
#    name = "diabetes_binary"
#)

#feature_data_pd = feature_dataset.df.toPandas()
# Drop the 'unique_id' column
#feature_data_pd = feature_data_pd.drop("unique_id", axis=1)


# Conforme a lógica original do exercício (Cell 10):
# Remover colunas de identificadores únicos se existirem no DataFrame
if 'unique_id' in feature_data_pd.columns:
    feature_data_pd = feature_data_pd.drop(columns=['unique_id'])
elif 'id' in feature_data_pd.columns:
    # Caso o dataset do Kaggle use 'id' minúsculo
    feature_data_pd = feature_data_pd.drop(columns=['id'])

# COMMAND ----------

display(feature_data_pd.tail())

# COMMAND ----------

import pandas as pd

# Convert all columns in the DataFrame to the 'double' data type
for column in feature_data_pd.columns:
    feature_data_pd[column] = feature_data_pd[column].astype("double")

# If you want to see the updated types
print(feature_data_pd.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train / Test Split
# MAGIC
# MAGIC Before proceeding with model training, it's essential to split the dataset into training and testing sets. This step ensures that the model is trained on one subset of the data and evaluated on an independent subset, providing a reliable estimate of its performance on new, unseen data.

# COMMAND ----------

from sklearn.model_selection import train_test_split

print(f"We have {feature_data_pd.shape[0]} records in our source dataset")

# split target variable into it's own dataset
target_col = "Diabetes_binary"
X_all = feature_data_pd.drop(labels=target_col, axis=1)
y_all = feature_data_pd[target_col]

# test / train split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=0.95, random_state=42)
print(f"We have {X_train.shape[0]} records in our training dataset")
print(f"We have {X_test.shape[0]} records in our test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Log the Model
# MAGIC
# MAGIC Now that we have our training and testing sets, let's fit a Decision Tree model to the training data. During this process, we will use MLflow to log various aspects of the model, including parameters, metrics, and the resulting model itself.

# COMMAND ----------

dtc_params = {
    'criterion': 'gini',
    'max_depth': 50,
    'min_samples_split': 20,
    'min_samples_leaf': 5
}

# COMMAND ----------

# MAGIC %md
# MAGIC In this code, we use MLflow to start a run and log parameters such as the criterion and max_depth of the Decision Tree model. After fitting the model on the training data, we evaluate its performance on the test set and log the accuracy as a metric.  
# MAGIC
# MAGIC 📌 Important: MLflow autologging is enabled by default on Databricks. This means you don't need to do anything for supported libraries. In the next section, we are disabling it and manually log params, metrics etc. just demonstrate how to do it manually when you need to log any custom model info.  
# MAGIC
# MAGIC 📌 Note: We won't define the experiment name, all runs generated in this notebook will be logged under the notebook title.

# COMMAND ----------

import mlflow

# Register models in UC
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

from math import sqrt

import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# set the path for mlflow experiment
#mlflow.set_experiment(f"/Users/{DA.username}/Demo-1.2-Model-Tracking-with-MLflow")
mlflow.set_experiment(f"/Workspace/Users/fabieneaulas@gmail.com/A3_Model_Tracking_With_MLflow_databrick_free")

# turn off autologging
mlflow.sklearn.autolog(disable=True)
#model_name = f"{DA.catalog_name}.{DA.schema_name}.diabetes-predictions"
model_name = f"diabetes-predictions"


# start an MLflow run
with mlflow.start_run(run_name="Model Tracking Demo") as run:
    # log the dataset
    feature_dataset = mlflow.data.from_pandas(feature_data_pd, name="diabetes_binary")
    mlflow.log_input(feature_dataset, context="source")
    mlflow.log_input(mlflow.data.from_pandas(X_train, source=feature_dataset.source), context="training")
    mlflow.log_input(mlflow.data.from_pandas(X_test, source=feature_dataset.source), context="test")

    # log our parameters
    mlflow.log_params(dtc_params)

    # fit our model
    dtc = DecisionTreeClassifier(**dtc_params)
    dtc_mdl = dtc.fit(X_train, y_train)

    # define model signature
    signature = infer_signature(X_all, y_all)

    # log the model
    mlflow.sklearn.log_model(
        sk_model = dtc_mdl,
        artifact_path = "model-artifacts",
        signature = signature,
        registered_model_name = model_name
    )

    # evaluate on the training set
    y_pred = dtc_mdl.predict(X_train)
    mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_pred))
    mlflow.log_metric("train_precision", precision_score(y_train, y_pred))
    mlflow.log_metric("train_recall", recall_score(y_train, y_pred))
    mlflow.log_metric("train_f1", f1_score(y_train, y_pred))

    # evaluate on the test set
    y_pred = dtc_mdl.predict(X_test)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred))
    mlflow.log_metric("test_f1", f1_score(y_test, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC At this point we can access all model details using the run.info class.

# COMMAND ----------

run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Model Artifacts
# MAGIC
# MAGIC In addition to logging parameters, metrics, and the model itself, we can also log artifacts—any files or data relevant to the run. Let's set up an MLflow client to log artifacts after the run is completed.

# COMMAND ----------

from mlflow.client import MlflowClient

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log Confusion Matrix
# MAGIC
# MAGIC The confusion matrix is a useful tool to visualize the classification performance of the model. It provides insights into the true positive, true negative, false positive, and false negative predictions.
# MAGIC
# MAGIC Let's create the confusion matrix and log it with MLflow using log_figure function.

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Computing the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[1, 0])

# Creating a figure object and axes for the confusion matrix
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting the confusion matrix using the created axes
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 0])
disp.plot(cmap=plt.cm.Blues, ax=ax)

# Setting the title of the plot
ax.set_title('Confusion Matrix')

# Now 'fig' can be used with MLflow's log_figure function
client.log_figure(run.info.run_id, figure=fig, artifact_file="A3_confusion_matrix.png")

# Showing the plot here for demonstration
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log Feature Importance
# MAGIC
# MAGIC Now, let's examine and log the resulting model. We'll extract and plot the feature importances inferred from the Decision Tree model to understand which data features are most critical for successful prediction.  
# MAGIC
# MAGIC Similar to the previous figure, we will use log_figure function.

# COMMAND ----------

import numpy as np

# Retrieving feature importances
feature_importances = dtc_mdl.feature_importances_
feature_names = X_train.columns.to_list()

# Plotting the feature importances
fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(feature_names))
ax.bar(y_pos, feature_importances, align='center', alpha=0.7)
ax.set_xticks(y_pos)
ax.set_xticklabels(feature_names, rotation=45)
ax.set_ylabel('Importance')
ax.set_title('Feature Importances in Decision Tree Classifier')

# log to mlflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="A3_feature_importances.png")

# display here
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can get the tree in text format or as a graph. To log the text format we will use log_artifact function.

# COMMAND ----------

print(f"The fitted DecisionTreeClassifier model has {dtc_mdl.tree_.node_count} nodes and is up to {dtc_mdl.tree_.max_depth} levels deep.")

# COMMAND ----------

# MAGIC %md
# MAGIC This is a very large decision tree, printing out the full tree logic, we can see it is vast and sprawling:

# COMMAND ----------

from sklearn.tree import export_text

text_representation = export_text(dtc_mdl, feature_names=feature_names)
print(text_representation)

# save this to a local file
tree_struct_filename = "A3_tree_structure.txt"
with open(tree_struct_filename, 'w') as f:
    f.write(text_representation)

# log it to mlflow
client.log_artifact(run.info.run_id, tree_struct_filename)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use dtreeviz to visualize the tree.
# MAGIC
# MAGIC dtreeviz is a python library for decision tree visualization and model interpretation.
# MAGIC
# MAGIC Let's use dtreeviz to create a tree structure and log it to MLflow using log_image function.

# COMMAND ----------

!pip install dtreeviz

# COMMAND ----------

from sklearn.tree import plot_tree

# Plot the tree structure (setting a reasonable max_depth for visualization)
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(
    dtc_mdl,
    max_depth=3,
    feature_names=feature_names,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    ax=ax
)

# Set title
ax.set_title("Decision Tree Visualization using plot_tree")

# log to mlflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="A3_plot_tree.png")

# display here
plt.show()

# COMMAND ----------

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create a tree visualization using sklearn's plot_tree (doesn't require Graphviz)
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    dtc_mdl,
    max_depth=3,
    feature_names=feature_names,
    class_names=['No Diabetes', 'Diabetes'],
    filled=True,
    ax=ax
)

# Set title
ax.set_title('Decision Tree Visualization (depth limited to 3 for readability)')

# Save the visualization to a local file
tree_viz_filename = "A3_0_tree_visualization.png"
fig.savefig(tree_viz_filename, dpi=150, bbox_inches='tight')

# Log it to mlflow
client.log_figure(run.info.run_id, figure=fig, artifact_file="tree_visualization.png")

# Display
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Depois clicar no filtro do databricks MLFlow

# COMMAND ----------

