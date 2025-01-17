# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import pickle
import os
import gzip
import json

# Paso 1. Cargar y preprocesar los datos
test_data = pd.read_csv("files/input/test_data.csv.zip")  # Leemos el dataframe de test_data.csv.zip
train_data = pd.read_csv("files/input/train_data.csv.zip")  # Leemos el dataframe de train_data.csv.zip

# Preprocesamiento para test_data
test_data.rename(columns={"default payment next month": "default"}, inplace=True)  # Renombramos la columna para test_data
test_data.drop(columns=["ID"], inplace=True)  # Eliminar la columna 'ID'
test_data.dropna(inplace=True)  # Eliminar filas con cualquier valor nulo
test_data["EDUCATION"] = test_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)  # Ajustar valores de EDUCATION
test_data = test_data.loc[test_data["MARRIAGE"] != 0]  # Filtrar registros donde MARRIAGE != 0
test_data = test_data.loc[test_data["EDUCATION"] != 0]  # Filtrar registros donde EDUCATION != 0

# Preprocesamiento para train_data
train_data.rename(columns={"default payment next month": "default"}, inplace=True)  # Renombramos la columna para train_data
train_data.drop(columns=["ID"], inplace=True)  # Eliminar la columna 'ID'
train_data.dropna(inplace=True)  # Eliminar filas con cualquier valor nulo
train_data["EDUCATION"] = train_data["EDUCATION"].apply(lambda x: 4 if x > 4 else x)  # Ajustar valores de EDUCATION
train_data = train_data.loc[train_data["MARRIAGE"] != 0]  # Filtrar registros donde MARRIAGE != 0
train_data = train_data.loc[train_data["EDUCATION"] != 0]  # Filtrar registros donde EDUCATION != 0

# Paso 2. Separar variables independientes y dependientes
X_train = train_data.drop(columns=["default"])  # Variables predictoras para entrenamiento
y_train = train_data["default"]  # Variable objetivo para entrenamiento

X_test = test_data.drop(columns=["default"])  # Variables predictoras para prueba
y_test = test_data["default"]  # Variable objetivo para prueba

# Paso 3. Definir las características categóricas y numéricas
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features = [col for col in X_train.columns if col not in categorical_features]

# Preprocesador con transformaciones para variables numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Escalado para variables numéricas
        ('cat', OneHotEncoder(), categorical_features)  # Codificación para variables categóricas
    ]
)

# Pipeline con selección de características 
pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ('feature_selection',SelectKBest(score_func=f_classif)),
        ('pca',PCA()),
        ('classifier',MLPClassifier(max_iter=15000,random_state=21))
        #('classifier',MLPClassifier(max_iter=10000,random_state=42))
    ]
)

# Paso 4. Definir la búsqueda de hiperparámetros para la optimización
param_grid = {
    'pca__n_components': [None],
    'feature_selection__k':[20],
    "classifier__hidden_layer_sizes": [(50, 30, 40,60)],
    'classifier__alpha': [0.26],
    "classifier__learning_rate_init": [0.001],
}
    

# Paso 5. Ejecutar la búsqueda de hiperparámetros con GridSearchCV
model = GridSearchCV(
    pipeline,
    param_grid,
    cv=10,  # Validación cruzada de 10 pliegues
    scoring="balanced_accuracy",  # Métrica de evaluación
    n_jobs=-1,  # Usar todos los núcleos disponibles
    refit=True
)

model.fit(X_train, y_train)  # Ajustar el modelo a los datos de entrenamiento

# Paso 6. Guardar el modelo entrenado
models_dir = 'files/models'
os.makedirs(models_dir, exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as file:
    pickle.dump(model, file)  # Guardar el modelo entrenado

# Paso 7. Evaluar el rendimiento en conjunto de entrenamiento y prueba
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_metrics = {
    "type": "metrics", # esto se agrega por el test
    'dataset': 'train',
    'precision': precision_score(y_train, y_train_pred),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred)
}


test_metrics = {
    "type": "metrics", # esto se agrega por el test
    'dataset': 'test',
    'precision': precision_score(y_test, y_test_pred),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_test_pred),
    'recall': recall_score(y_test, y_test_pred),
    'f1_score': f1_score(y_test, y_test_pred)
}
output_file = "files/output/metrics.json"
os.makedirs("files/output", exist_ok=True)

# Guardar métricas de evaluación en un archivo JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(train_metrics, f, ensure_ascii=False) #indent=4
    f.write('\n')
    json.dump(test_metrics, f, ensure_ascii=False) #indent=4
    f.write('\n')


# Paso 8. Generar matrices de confusión
train_cm = confusion_matrix(y_train, y_train_pred)
test_cm = confusion_matrix(y_test, y_test_pred)

train_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'train',
    'true_0': {'predicted_0': int(train_cm[0, 0]), 'predicted_1': int(train_cm[0, 1])},
    'true_1': {'predicted_0': int(train_cm[1, 0]), 'predicted_1': int(train_cm[1, 1])}
}

test_cm_dict = {
    'type': 'cm_matrix',
    'dataset': 'test',
    'true_0': {'predicted_0': int(test_cm[0, 0]), 'predicted_1': int(test_cm[0, 1])},
    'true_1': {'predicted_0': int(test_cm[1, 0]), 'predicted_1': int(test_cm[1, 1])}
}

output_path = 'files/output/metrics.json'

with open(output_path, 'a', encoding='utf-8') as f:
    json.dump(train_cm_dict, f, ensure_ascii=False) 
    f.write('\n')
    json.dump(test_cm_dict, f, ensure_ascii=False)  
    f.write('\n')