#dataset obtenido de:https://datos.gob.mx/busca/dataset/indicadores-de-pobreza-pobreza-por-ingresos-rezago-social-y-gini-a-nivel-municipal1990-200-2010
#diccionario: https://www.coneval.org.mx/Informes/Pobreza/Datos_abiertos/Indicadores_municipales/Indicadores_municipales_sabana_DIC.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('/media/eduardo/SSD Kingston Datos/Projects/machine_learning/pobreza/Indicadores_municipales_sabana_DA.csv', 
                 index_col=0, sep=',', encoding='latin-1')

#se analizan las columnas y se muestran las que tienen datos faltantes
rows_with_null = df[df.isnull().any(axis=1)]
#print(rows_with_null)

df.bfill(inplace=True)
df.ffill(inplace=True)

data = df[df['nom_ent'] == 'Zacatecas']
print(data)
#print(df.columns)

plt.subplot(1,2,1)
plt.title('N vul ing')
plt.hist(data['N_vul_ing'],edgecolor ='black')

plt.subplot(1,2,2)
plt.boxplot(data['N_vul_ing'],vert=False)

plt.tight_layout() 
plt.show()