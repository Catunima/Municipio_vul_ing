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
#?identificacion de valores para rangos de niveles
vul_row = len(vul_ing)

first = vul_ing.head(vul_row // 2)
last = vul_ing.tail(vul_row // 2)

first_limit = 181621
second_limit = 60000
ct = 0

# Lista de sufijos de años a eliminar
years = ["00","05","10","_00", "_05", "_10"]

# Filtra las columnas que no terminan con los sufijos de años y crea un nuevo DataFrame
new_df = df[[col for col in df.columns if not col.endswith(tuple(years))]]

# Ahora df_sin_años contiene solo las columnas que no están relacionadas con años
new_df = new_df.drop(columns=['nom_ent','clave_mun','mun',''])
new_df['has_vul'] = 0

while ct != len(new_df['N_vul_ing']):
    value = df['N_vul_ing'].iloc[ct]
    if value >= second_limit:
        new_df['has_vul'].iloc[ct] = 1
    ct += 1

new_df['yes'] = (new_df['has_vul'] == 1).astype(int)
new_df['no'] = (new_df['has_vul'] == 0).astype(int)

print(new_df[['N_vul_ing','has_vul','yes','no']].head(15))
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