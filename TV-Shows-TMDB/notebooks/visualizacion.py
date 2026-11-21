#semana 6

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def guardar_grafica(nombre):
    ruta = Path(__file__).resolve().parent.parent / "resultados" / "graficas"
    ruta.mkdir(parents=True, exist_ok=True)
    plt.savefig(ruta / f"{nombre}.png", dpi=300, bbox_inches="tight")

    print(f"\nGrafica guardada: {nombre}.png")

def grafica_histograma(df):
    plt.figure(figsize=(10, 5))

    sns.histplot(data=df, x="popularity", bins=30, kde=True)

    plt.title("Distribucion de Popularidad de las Series")
    plt.xlabel("Popularidad")
    plt.ylabel("Cantidad de Series")    

    plt.tight_layout()

    guardar_grafica("histograma_PopularidadPorSerie")

    plt.show()

def grafica_caja(df):
    plt.figure(figsize=(8, 5))

    sns.boxplot(data=df, y="vote_average")

    plt.title("Calificaciones")
    plt.ylabel("Promedio de Votos")

    plt.tight_layout()

    guardar_grafica("caja_CalificacionesVotos")

    plt.show()

def grafica_dispersion(df):
    plt.figure(figsize=(10, 5))

    sns.scatterplot(data=df, x="popularity", y="vote_average", alpha=0.5)

    plt.title("Popularidad vs Promedio de Votos")
    plt.xlabel("Popularidad")
    plt.ylabel("Promedio de votos")

    plt.tight_layout()

    guardar_grafica("dispersion_PopularidadVotos")

    plt.show()

def grafica_pastel(df):
    idiomas = df["original_language"].value_counts().head(8)

    plt.figure(figsize=(8, 8))

    plt.pie(idiomas.values, labels=idiomas.index, autopct="%1.1f%%", startangle=90)

    plt.title("Porcentaje de Series por Idioma")

    plt.tight_layout()

    guardar_grafica("pastel_IdiomasPorcentaje")

    plt.show()

def grafica_barras(df):
    idiomas = df["original_language"].value_counts().head(10)

    plt.figure(figsize=(10, 5))

    sns.barplot(x=idiomas.index, y=idiomas.values)

    plt.title("Series por Idioma Original")
    plt.xlabel("Idioma Original")
    plt.ylabel("Cantidad de Series")

    plt.tight_layout()

    guardar_grafica("barras_IdiomaCantidad")

    plt.show()

def grafica_lineas(df):
    fechas = pd.to_datetime(df["first_air_date"], errors="coerce")

    series_por_anio = fechas.dt.year.value_counts().sort_index()

    plt.figure(figsize=(12, 5))

    plt.plot(series_por_anio.index, series_por_anio.values, marker="o")

    plt.title("Cantidad de Series Estrenadas por Año")
    plt.xlabel("Año")
    plt.ylabel("Cantidad de series")

    plt.tight_layout()

    guardar_grafica("lineas_EstrenosPorAño")

    plt.show()

def menu_visualizacion(df):

    graficas = {
        "1": ("Distribucion de Popularidad de las Series", grafica_histograma),
        "2": ("Promedio de Votos por Clasificacion", grafica_caja),
        "3": ("Popularidad vs Promedio de Votos", grafica_dispersion),
        "4": ("Distribucion de Series por Idiomas (Porcentaje)", grafica_pastel),
        "5": ("Distribucion de Series por Idioma (Cantidad)", grafica_barras),
        "6": ("Cantidad de Series Estrenadas por Año", grafica_lineas)
    }

    while True:

        print("\n===================================")
        print("       VISUALIZACION DE DATOS")
        print("===================================")

        for opcion, (nombre, funcion) in graficas.items():
            print(f"{opcion}. {nombre}")

        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "0":
            break

        elif opcion in graficas:
            nombre, funcion = graficas[opcion]
            print(f"\nGenerando: {nombre}...")
            funcion(df)
        else:
            print("\nOpcion no valida...")