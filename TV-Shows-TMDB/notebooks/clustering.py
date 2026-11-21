#semana 12
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def preparar_datos(df):
    datos = df[["name", "popularity", "vote_average", "vote_count"]].dropna()

    datos = datos[(datos["popularity"] >= 0) & (datos["vote_average"] >= 0) & (datos["vote_count"] >= 0)].copy()

    if len(datos) < 10:
        print("\nNo hay suficientes datos para realizar clustering.")
        return None

    X = datos[["popularity", "vote_average", "vote_count"]]

    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(X)

    return X_escalado, datos, escalador

def entrenar_modelo(X_escalado, k=3):
    modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    etiquetas = modelo.fit_predict(X_escalado)
    return modelo, etiquetas

def determinar_k(X_escalado):
    inercias = []
    valores_k = range(1, min(10, len(X_escalado)) + 1)

    for k in valores_k:
        modelo = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        modelo.fit(X_escalado)
        inercias.append(modelo.inertia_)

    return valores_k, inercias

def graficar_codo(valores_k, inercias):
    carpeta = obtener_carpeta_resultados()

    plt.figure(figsize=(8, 5))
    plt.plot(valores_k, inercias, marker="o")

    plt.title("Metodo del codo - K-Means")
    plt.xlabel("Numero de clusters (K)")
    plt.ylabel("Inercia")
    plt.xticks(list(valores_k))
    plt.grid(True)

    ruta = carpeta / "metodo_codo.png"
    plt.savefig(ruta)
    print(f"\nGrafica del metodo del codo guardada corractamente")
    plt.show()
    plt.close()

def evaluar_modelo(modelo, X_escalado, etiquetas):
    carpeta = obtener_carpeta_resultados()

    inercia = modelo.inertia_
    if len(set(etiquetas)) < 2 or len(set(etiquetas)) >= len(etiquetas):
        print("\nNo se puede calcular el coeficiente Silhouette.")
        return None

    silhouette = silhouette_score(X_escalado, etiquetas)

    cantidad_clusters = len(set(etiquetas))

    print("\n===== EVALUACION DEL MODELO K-MEANS =====")
    print(f"Numero de clusters: {cantidad_clusters}")
    print(f"Inercia: {inercia:.2f}")
    print(f"Coeficiente Silhouette: {silhouette:.4f}")

    print("\nDistribucion de series por cluster:")

    for cluster in range(cantidad_clusters):
        cantidad = np.sum(etiquetas == cluster)
        print(f"Cluster {cluster}: {cantidad} series")

    evaluacion = pd.DataFrame({"Clusters": [cantidad_clusters], "Inercia": [inercia], "Silhouette": [silhouette]})

    evaluacion.to_csv(carpeta / "evaluacion_clustering.csv", index=False)
    print(f"\nEvaluacion guardada en: {(carpeta / 'evaluacion_clustering.csv').resolve()}")

    return evaluacion

def graficar_clusters(datos, etiquetas, modelo, escalador):
    carpeta = obtener_carpeta_resultados()

    datos_grafica = datos.copy()
    datos_grafica["cluster"] = etiquetas
    centroides_originales = escalador.inverse_transform(modelo.cluster_centers_)

    plt.figure(figsize=(9, 6))

    plt.scatter(datos_grafica["popularity"], datos_grafica["vote_average"], c=datos_grafica["cluster"], cmap="viridis", alpha=0.7)
    plt.scatter(centroides_originales[:, 0], centroides_originales[:, 1], marker="X", s=250, color="red", label="Centroides")

    plt.title("Clusters de series de television")
    plt.xlabel("Popularidad")
    plt.ylabel("Calificacion promedio")
    plt.colorbar(label="Cluster")
    plt.legend()
    plt.grid(True)

    ruta = carpeta / "clusters.png"
    plt.savefig(ruta)
    print(f"\nGrafica de clusters guardada correctamente")
    plt.show()
    plt.close()

def guardar_resultados(datos, etiquetas):
    carpeta = obtener_carpeta_resultados()

    resultados = datos.copy()
    resultados["cluster"] = etiquetas

    resultados.to_csv(carpeta / "resultados_clustering.csv", index=False)

    print(f"\nResultados guardados en: {(carpeta / 'resultados_clustering.csv').resolve()}")
    print(f"Total de series agrupadas: {len(resultados)}")

    print("\nCantidad de series por cluster:")
    print(resultados["cluster"].value_counts().sort_index())

def solicitar_k():
    try:
        k = int(input("\nIngrese el numero de clusters: "))
        return k
    except ValueError:
        print("\nDebe ingresar un numero entero.")
        return None

def obtener_carpeta_resultados():
    base_dir = Path(__file__).resolve().parent.parent
    carpeta = base_dir / "resultados" / "modelos" / "clustering"
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta

def menu_clustering(df):
    datos_preparados = preparar_datos(df)

    if datos_preparados is None:
        return

    X_escalado, datos, escalador = datos_preparados

    modelo = None
    etiquetas = None

    while True:
        print("\n===== CLUSTERING - K-MEANS =====")
        print("1. Determinar numero de clusters (Metodo del codo)")
        print("2. Entrenar modelo K-Means")
        print("3. Evaluar modelo")
        print("4. Graficar clusters")
        print("5. Guardar resultados")
        print("6. Ejecutar analisis completo")
        print("0. Regresar al menu principal")

        opcion = input("\nSeleccione una opcion: ")

        if opcion == "1":
            valores_k, inercias = determinar_k(X_escalado)
            graficar_codo(valores_k, inercias)

        elif opcion == "2":
            k = solicitar_k()
            if k is None:
                continue
            if k < 2 or k > min(10, len(datos) - 1):
                print("\nEl numero de clusters no es valido. Utilice un valor entre 2 y 10.")
                continue
            modelo, etiquetas = entrenar_modelo(X_escalado, k)
            print("\nModelo K-Means entrenado correctamente.")

        elif opcion == "3":
            if modelo is None:
                print("\nPrimero debe entrenar el modelo.")
            else:
                evaluar_modelo(modelo, X_escalado, etiquetas)

        elif opcion == "4":
            if etiquetas is None:
                print("\nPrimero debe entrenar el modelo.")
            else:
                graficar_clusters(datos, etiquetas, modelo, escalador)

        elif opcion == "5":
            if etiquetas is None:
                print("\nPrimero debe entrenar el modelo.")
            else:
                guardar_resultados(datos, etiquetas)

        elif opcion == "6":
            valores_k, inercias = determinar_k(X_escalado)
            graficar_codo(valores_k, inercias)

            k = solicitar_k()
            if k is None:
                continue
            if k < 2 or k > min(10, len(datos) - 1):
                print("\nEl numero de clusters no es valido. Utilice un valor entre 2 y 10.")
                continue

            modelo, etiquetas = entrenar_modelo(X_escalado, k)

            evaluar_modelo(modelo, X_escalado, etiquetas)
            graficar_clusters(datos, etiquetas, modelo, escalador)
            guardar_resultados(datos, etiquetas)

        elif opcion == "0":
            break

        else:
            print("\nOpcion no valida.")