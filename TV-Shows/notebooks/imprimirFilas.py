import pandas as pd

df = pd.read_csv("../BDatos/10k_Poplar_Tv_shows.csv")

print("Filas:", df.shape)
print("Columnas:", df.columns.tolist())

print("\nColumnas:")
print(df.columns.tolist())

print("\nInformación:")
print(df.info())

print("\nPrimeras filas:")
print(df.head())