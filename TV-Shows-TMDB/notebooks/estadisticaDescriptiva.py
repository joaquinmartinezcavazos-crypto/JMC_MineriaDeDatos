#semana 5

import pandas as pd

def resumen_general(df):
    print("\n===================================")
    print("         RESUMEN GENERAL")
    print("===================================")

    print(f"\nTotal de registros: {df.shape[0]}")
    print(f"Total de columnas: {df.shape[1]}")

    print("\nColumnas del dataset:")
    print(df.columns.tolist())

    print("\nTipos de datos:")
    print(df.dtypes)

    print("\nValores nulos por columna:")
    print(df.isnull().sum())

    print("\nRegistros duplicados:")
    print(df.duplicated().sum())

def analizar_numericas(df):
    columnas = ["popularity", "vote_average", "vote_count"]

    print("\n===================================")
    print("       ESTADISTICAS NUMERICAS")
    print("===================================")

    for columna in columnas:
        print(f"\nVariable: {columna}")
        print("-----------------------------------\n")

        datos = df[columna].dropna()

        print(f"Cantidad de datos: {len(datos)}")
        print(f"Media: {datos.mean():.2f}")
        print(f"Mediana: {datos.median():.2f}")
        print(f"Moda: {datos.mode().tolist()}")
        print(f"Minimo: {datos.min():.2f}")
        print(f"Maximo: {datos.max():.2f}")
        print(f"Rango: {datos.max() - datos.min():.2f}")
        print(f"Varianza: {datos.var():.2f}")
        print(f"Desviacion estandar: {datos.std():.2f}")

        print(f"Primer cuartil: {datos.quantile(0.25):.2f}")
        print(f"Segundo cuartil: {datos.quantile(0.50):.2f}")
        print(f"Tercer cuartil: {datos.quantile(0.75):.2f}")

def analizar_categoricas(df):
    columnas = ["adult", "original_language", "origin_country"]

    print("\n===================================")
    print("      VARIABLES CATEGORICAS")
    print("===================================")

    for columna in columnas:

        print(f"\nVariable: {columna}")
        print("-----------------------------------")

        frecuencias = df[columna].value_counts()

        print(f"Categorias diferentes: {df[columna].nunique()}")

        print("\nFrecuencias:")
        print(frecuencias.head(10).to_string())

        print("\nPorcentajes:")
        porcentajes = df[columna].value_counts(normalize=True) * 100
        print(porcentajes.head(10).round(2).to_string())

def analizar_fechas(df):
    print("\n===================================")
    print("          ANALISIS DE FECHAS")
    print("===================================")

    fechas = pd.to_datetime(df["first_air_date"], errors="coerce").dropna()

    if fechas.empty:
        print("\nNo hay fechas validas para analizar.")
        return

    print(f"\nFechas validas: {len(fechas)}")
    print(f"Fecha mas antigua: {fechas.min().date()}")
    print(f"Fecha mas reciente: {fechas.max().date()}")

    print("\nSeries estrenadas por año:")

    por_anio = fechas.dt.year.value_counts().sort_index()

    print(por_anio.to_string())

def menu_estadistica_descriptiva(df):
    while True:
        print("\n===================================")
        print("       ESTADISTICA DESCRIPTIVA")
        print("===================================")
        print("1. Resumen general del dataset")
        print("2. Estadisticas numericas")
        print("3. Analisis de variables categoricas")
        print("4. Analisis de fechas")
        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opcion: ")

        if opcion == "1":
            resumen_general(df)
        elif opcion == "2":
            analizar_numericas(df)
        elif opcion == "3":
            analizar_categoricas(df)
        elif opcion == "4":
            analizar_fechas(df)
        elif opcion == "0":
            break
        else:
            print("\nOpcion no valida...")