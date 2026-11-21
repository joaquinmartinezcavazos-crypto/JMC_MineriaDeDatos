print("Iniciando el programa...")
import pandas as pd
from pathlib import Path

from notebooks.imprimirFilas import menu_imprimir_filas
from notebooks.estadisticaDescriptiva import menu_estadistica_descriptiva
from notebooks.datosAgrupados import menu_datos_agrupados
from notebooks.visualizacion import menu_visualizacion
from notebooks.pruebasEstadisticas import menu_pruebas_estadisticas
from notebooks.modelosLineales import menu_modelos_lineales
from notebooks.clasificacion import menu_clasificacion
from notebooks.clustering import menu_clustering
from notebooks.pronostico import menu_pronostico
from notebooks.analisisTexto import menu_analisis_texto

def cargar_dataset():
    ruta = Path(__file__).resolve().parent / "BDatos" / "10k_Poplar_Tv_Shows.csv"
    return pd.read_csv(ruta)

def menu_principal():
    df = cargar_dataset()

    while True:
        print("\n\n===================================")
        print("       ANALISIS DE DATOS TMDB")
        print("===================================")
        print("1. Imprimir filas")
        print("2. Estadistica descriptiva")
        print("3. Datos agrupados")
        print("4. Visualizacion de datos")
        print("5. Pruebas estadisticas")
        print("6. Modelos lineales")
        print("7. Clasificacion de datos (KNN)")
        print("8. Clustering K-Means")
        print("9. Pronostico (Forecasting)")
        print("10. Analisis de texto (Word Cloud)")
        print("0. Salir")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            menu_imprimir_filas(df)

        elif opcion == "2":
            menu_estadistica_descriptiva(df)

        elif opcion == "3":
            menu_datos_agrupados(df)

        elif opcion == "4":
            menu_visualizacion(df)

        elif opcion == "5":
            menu_pruebas_estadisticas(df)

        elif opcion == "6":
            menu_modelos_lineales(df)

        elif opcion == "7":
            menu_clasificacion(df)

        elif opcion == "8":
            menu_clustering(df)

        elif opcion == "9":
            menu_pronostico(df)

        elif opcion == "10":
            menu_analisis_texto(df)

        elif opcion == "0":
            print("\nFinalizando programa...")
            break

        else:
            print("\nOpcion no valida...")

menu_principal()