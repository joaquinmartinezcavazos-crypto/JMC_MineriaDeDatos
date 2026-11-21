#semana 5

import pandas as pd
import numpy as np
import math
from pathlib import Path

def crear_tabla_frecuencias(df, columna, intervalos=None):
    datos = df[columna].dropna()
    if datos.empty:
        print("\nNo hay datos disponibles para analizar.")
        return None

    cantidad = len(datos)

    #cantidad de intervalos
    if intervalos is None:
        intervalos = math.ceil(1 + 3.322 * math.log10(cantidad))

    #intervalos
    frecuencias, limites = np.histogram(datos, bins=intervalos)

    #marcas de clase
    marcas = (limites[:-1] + limites[1:]) / 2

    #frecuencias acumuladas
    frecuencia_acumulada = np.cumsum(frecuencias)

    #frecuencias relativas
    frecuencia_relativa = frecuencias / cantidad

    #frecuencias relativas acumuladas
    relativa_acumulada = np.cumsum(frecuencia_relativa)

    tabla = pd.DataFrame({
        "Intervalo": [f"{limites[i]:.2f} - {limites[i + 1]:.2f}" for i in range(len(frecuencias))],
        "Marca de clase": marcas,
        "Frecuencia absoluta": frecuencias,
        "Frecuencia acumulada": frecuencia_acumulada,
        "Frecuencia relativa": frecuencia_relativa,
        "Frecuencia relativa acumulada": relativa_acumulada})

    return tabla, marcas, frecuencias, cantidad

def calcular_metricas(marcas, frecuencias, cantidad):

    media = np.sum(marcas * frecuencias) / cantidad

    varianza = np.sum(frecuencias * (marcas - media) ** 2) / cantidad

    desviacion = np.sqrt(varianza)

    rango = np.max(marcas[frecuencias > 0]) - np.min(marcas[frecuencias > 0])

    print("\n===================================")
    print("       METRICAS AGRUPADAS")
    print("===================================")

    print(f"Media agrupada: {media:.2f}")
    print(f"Varianza agrupada: {varianza:.2f}")
    print(f"Desviacion estandar agrupada: {desviacion:.2f}")
    print(f"Rango agrupado: {rango:.2f}")

def analizar_variable(df, columna):
    resultado = crear_tabla_frecuencias(df, columna)

    if resultado is None:
        return

    tabla, marcas, frecuencias, cantidad = resultado

    print("\n===================================")
    print(f"    DATOS AGRUPADOS: {columna}")
    print("===================================")

    print(f"\nCantidad de observaciones: {cantidad}")

    print("\nTabla de frecuencias:\n")
    print(tabla.to_string(index=False))

    ruta = Path(__file__).resolve().parent.parent / "resultados" / "estadistica" / "datos_agrupados"
    ruta.mkdir(parents=True, exist_ok=True)
    
    tabla.to_csv(ruta / f"tabla_frecuencias_{columna}.csv", index=False, encoding="utf-8-sig")

    print("\nTabla guardada correctamente.")

    calcular_metricas(marcas, frecuencias, cantidad)

def menu_datos_agrupados(df):

    while True:
        print("\n===================================")
        print("          DATOS AGRUPADOS")
        print("===================================")
        print("1. Agrupar popularidad")
        print("2. Agrupar promedio de votos")
        print("3. Agrupar cantidad de votos")
        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            analizar_variable(df, "popularity")
        elif opcion == "2":
            analizar_variable(df, "vote_average")
        elif opcion == "3":
            analizar_variable(df, "vote_count")
        elif opcion == "0":
            break
        else:
            print("\nOpcion no valida...")