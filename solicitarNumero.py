def solicitar_numero():
    while True:
        try:
            numero = int(input("Ingresa un número entre 1 y 100: "))
            if 1 <= numero <= 100:
                print(f"Gracias! Has ingresado el número {numero}.")
                break
            else:
                print("El número debe estar entre 1 y 100. Inténtalo de nuevo.")
        except ValueError:
            print("Entrada no válida. Por favor, ingresa un número.")

solicitar_numero()
