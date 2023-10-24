#dataset obtenido de:https://datos.gob.mx/busca/dataset/indicadores-de-pobreza-pobreza-por-ingresos-rezago-social-y-gini-a-nivel-municipal1990-200-2010
#diccionario: https://www.coneval.org.mx/Informes/Pobreza/Datos_abiertos/Indicadores_municipales/Indicadores_municipales_sabana_DIC.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import Perceptron

from sklearn.model_selection import train_test_split


df = pd.read_csv('/media/eduardo/SSD Kingston Datos/Projects/machine_learning/pobreza/Indicadores_municipales_sabana_DA.csv', 
                 index_col=0, sep=',', encoding='latin-1')

#se analizan las columnas y se muestran las que tienen datos faltantes
#!eliminacion de nulos
rows_with_null = df[df.isnull().any(axis=1)]
#print(rows_with_null)

df.bfill(inplace=True)
df.ffill(inplace=True)

#!seleccion y orden de columna
vul_ing = df.sort_values(by='N_vul_ing',ascending=False)#ordenamiento de valores en descendente
#?identificacion de valores para rangos de niveles
vul_row = len(vul_ing)

first = vul_ing.head(vul_row // 2)
last = vul_ing.tail(vul_row // 2)

first_limit = 181621
second_limit = 60000
ct = 0

# Lista de sufijos de años a eliminar
years = ["90","00","05","10","_00", "_05", "_10", "_90"]

# Filtra las columnas que no terminan con los sufijos de años y crea un nuevo DataFrame
new_df = df[[col for col in df.columns if not col.endswith(tuple(years))]]

# Ahora df_sin_años contiene solo las columnas que no están relacionadas con años
new_df = new_df.drop(columns=['nom_ent','clave_mun','mun','nom_mun'])
new_df['has_vul'] = 0

while ct != len(new_df['N_vul_ing']):
    value = df['N_vul_ing'].iloc[ct]
    if value >= second_limit:
        new_df['has_vul'].iloc[ct] = 1
    ct += 1

new_df['yes'] = (new_df['has_vul'] == 1).astype(int)
new_df['no'] = (new_df['has_vul'] == 0).astype(int)

#print(new_df[['N_vul_ing','has_vul','yes','no']].head(15))
# Lista de columnas en el orden deseado (sin "has_vul")
columnas_ordenadas = [col for col in new_df.columns if col != 'has_vul']

# Agrega 'has_vul' al final de la lista de columnas
columnas_ordenadas.append('has_vul')

# Crea un nuevo DataFrame con las columnas en el orden deseado
dataframe = new_df[columnas_ordenadas]

#!Separacion de datos de datafram por x y y para entrenamiento de modelos
# Dividir el DataFrame en entrenamiento (80%) y prueba (20%)
'''train_size = int(0.8 * len(dataframe))
train_data = dataframe[:train_size]
test_data = dataframe[train_size:]

# Separar las características (X) y la variable objetivo (y)
X_train = train_data.drop(columns=['has_vul'])  # Quita la columna 'has_vul' de las características
y_train = train_data['has_vul']  # Variable objetivo

X_test = test_data.drop(columns=['has_vul'])  # Quita la columna 'has_vul' de las características
y_test = test_data['has_vul']  # Variable objetivo'''
X = dataframe.drop(columns=['has_vul'])
y = dataframe['has_vul']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#!implementacion de algoritmo de knn

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Calcula las distancias entre x y los puntos en el conjunto de entrenamiento
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Ordena por distancia y devuelve las etiquetas de los primeros k vecinos
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Devuelve la etiqueta más común entre los k vecinos
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common

# Usar X_train e y_train de tu conjunto de datos
knn = KNN(k=3)
knn.fit(X_train.values, y_train.values)
predictions = knn.predict(X_test.values)
print("Prediction with algorith")
print(f'Predicción: {predictions}, Etiqueta real: {y_test.values}')
# Puedes comparar tus predicciones con y_test para evaluar el rendimiento

#!entrenamiendo de modelo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# prediction
prediction_knn = knn.predict(X_test)
print("Prediction with library")
print("Prediction for test set: {}".format(prediction_knn))
#Actual value and the predicted value
a = pd.DataFrame({'Actual value': y_test, 'Predicted value': prediction_knn})
print(a.tail(20))

#Evaluar modelo
matrix = confusion_matrix(y_test, prediction_knn)
sns.heatmap(matrix, annot=True, fmt="d")
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
print(classification_report(y_test, prediction_knn))
#! algoritmo de perceptron
print('perceptron con algoritmo')
class MiPerceptron:
    def __init__(self, num_features):
        # Inicializa los pesos y el sesgo (bias) de forma aleatoria
        self.weights = np.random.rand(num_features)
        self.bias = np.random.rand()
        
    def predict(self, inputs):
        # Realiza la suma ponderada de las entradas y aplica la función de paso
        linear_sum = self.bias + np.dot(self.weights, inputs)
        return 1 if linear_sum >= 0 else 0
    
    def train(self, X, y, learning_rate=0.1, epochs=100):
        for _ in range(epochs):
            for i in range(X.shape[0]):
                inputs = X[i]
                target = y[i]
                prediction = self.predict(inputs)
                error = target - prediction
                # Actualiza los pesos y el sesgo
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

# Supongamos que tienes tus datos de entrenamiento en X_train y y_train
# Asegúrate de que los datos estén en formato NumPy
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()

# Crea un perceptrón y entrénalo con los datos de entrenamiento
mi_perceptron = MiPerceptron(num_features=X_train.shape[1])
mi_perceptron.train(X_train, y_train)

# Ahora puedes hacer predicciones en tus datos de prueba
# Supongamos que tienes tus datos de prueba en X_test
X_test = X_test.to_numpy()
y_pred = [mi_perceptron.predict(inputs) for inputs in X_test]
print("Predicciones en los datos de prueba:")
print(y_pred)
# Luego, puedes evaluar el rendimiento del modelo, por ejemplo, calculando la precisión

accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo:", accuracy)

#!perceptron
print('perceptron library')
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X_train,y_train)
Perceptron()
print(clf.predict(X_test))