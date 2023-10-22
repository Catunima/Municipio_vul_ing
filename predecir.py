#dataset obtenido de:https://datos.gob.mx/busca/dataset/indicadores-de-pobreza-pobreza-por-ingresos-rezago-social-y-gini-a-nivel-municipal1990-200-2010
#diccionario: https://www.coneval.org.mx/Informes/Pobreza/Datos_abiertos/Indicadores_municipales/Indicadores_municipales_sabana_DIC.txt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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

vul_row = len(vul_ing)
quartile = vul_ing.head(vul_row // 4)
print(quartile['N_vul_ing'].max())
#division de columnas por partes
#df['very_high'] = (vul_ing.index < quartile).astype(int)
#df['high'] = (quartile <= vul_ing.index) & (vul_ing.index < 2 * quartile).astype(int)
#df['medium'] = (2 * quartile <= vul_ing.index) & (vul_ing.index < 3 * quartile).astype(int)
#df['low'] = (3 * quartile <= vul_ing.index) & (vul_ing.index < 4 * quartile).astype(int)
##!limpieza de outliers
#plt.figure(figsize=(10, 6))
"""sns.snsplot(data=df,kind="bar", height=7, aspect=2)
plt.xticks(rotation=90)
plt.show()
"""
#data = df[df['N_vul_ing']]
#print(data)
#print(df.columns)
""""
plt.subplot(1,2,1)
plt.title('N vul ing')
plt.hist(data['N_vul_ing'],edgecolor ='black')

plt.subplot(1,2,2)
plt.boxplot(data['N_vul_ing'],vert=True)

plt.tight_layout() 
plt.show()"""