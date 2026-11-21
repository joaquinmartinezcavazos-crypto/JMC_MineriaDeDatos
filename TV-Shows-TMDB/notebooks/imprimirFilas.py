#semana 1

def imprimir_filas(df, cantidad=5, tail =False, inicio=1):

    if cantidad <= 0:
        print("\nLa cantidad de filas enviadas fue 0 o menor...")
        return

    if tail:
        cantidad = min(cantidad, len(df))

        print(f"\nUltimas {cantidad} filas:\n")
        print(df.tail(cantidad).to_string(index=False))
        return

    inicio -= 1

    if inicio < 0 or inicio >= len(df):
        print("\nEl indice inicial no es valido.")
        return
    
    
    fin = min(inicio + cantidad, len(df))

    print(f"\nMostrando filas {inicio + 1} a {fin}:\n")
    print(df.iloc[inicio:fin].to_string(index=False))


def menu_imprimir_filas(df):

    while True:
        print("\n===================================")
        print("           IMPRIMIR FILAS")
        print("===================================")
        print(f"Registros disponibles: {len(df)}")
        print("1. Imprimir primeras filas")
        print("2. Imprimir filas personalizadas")
        print("3. Imprimir ultimas filas")
        print("0. Regresar al menu principal")
        print("===================================")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            try:
                cantidad = int(input("Inserte cuantas filas deseas imprimir: "))
                imprimir_filas(df, cantidad=cantidad)
            except ValueError:
                print("\nIntroduzca un numero entero valido.")

        elif opcion == "2":
            try:
                cantidad = int(input("Inserte cuantas filas deseas imprimir: "))
                inicio = int(input("Inserte la fila inicial: "))
                imprimir_filas(df, cantidad=cantidad, inicio=inicio)
            except ValueError:
                print("\nIntroduzca numeros enteros valido.")

        elif opcion == "3":
            try:
                cantidad = int(input("Inserte cuantas filas deseas imprimir: "))
                imprimir_filas(df, cantidad=cantidad, tail=True)
            except ValueError:
                print("\nIntroduzca un numero entero valido.")

        elif opcion == "0":
            break
        else:
            print("\nOpcion no valida...")