# Introducción a Listas en Python

# Definición y características de las listas
# Las listas en Python son colecciones ordenadas de elementos que pueden ser de diferentes tipos.
# Son mutables, lo que significa que se pueden modificar después de su creación.

# Creación de listas
mi_lista = []  # Lista vacía
mi_lista = [1, 2, 3, "cuatro", 5.0, True]  # Lista con elementos de diferentes tipos

# Acceso a elementos por índice
print(mi_lista[0])  # Acceder al primer elemento (índice 0) -> 1
print(mi_lista[-1])  # Acceder al último elemento (índice -1) -> True

# Métodos básicos de las listas
mi_lista.append("nuevo_elemento")  # Agregar un elemento al final de la lista
print(mi_lista)

mi_lista.insert(2, "insertado")  # Insertar un elemento en una posición específica
print(mi_lista)

mi_lista.remove("cuatro")  # Eliminar un elemento específico
print(mi_lista)

ultimo_elemento = mi_lista.pop()  # Eliminar el último elemento y retornarlo
print(ultimo_elemento)
print(mi_lista)

indice = mi_lista.index(5.0)  # Encontrar el índice de un elemento
print(indice)

# Operaciones con listas

# Concatenación de listas
lista1 = [1, 2, 3]
lista2 = [4, 5, 6]
lista_concatenada = lista1 + lista2
print(lista_concatenada)

# Slicing (rebanado) de listas
sublista = lista_concatenada[0:3]  # Obtener una sublista del primer al tercer elemento (excluye el índice 3)
print(sublista)

sublista = lista_concatenada[1:]  # Obtener una sublista desde el segundo elemento hasta el final
print(sublista)

sublista = lista_concatenada[::2]  # Obtener una sublista con elementos alternos
print(sublista)

# Longitud de una lista
longitud = len(lista_concatenada)
print(longitud)

# Ordenamiento de listas
lista_desordenada = [3, 1, 4, 2, 5]

# Ordenar sin modificar la lista original
lista_ordenada = sorted(lista_desordenada)
print(lista_ordenada)

# Ordenar modificando la lista original
lista_desordenada.sort()
print(lista_desordenada)

# Comprensión de listas
# Sintaxis básica
cuadrados = [x**2 for x in range(1, 6)]  # Crear una lista de cuadrados de números del 1 al 5
print(cuadrados)

# Uso de condicionales
pares = [x for x in range(1, 11) if x % 2 == 0]  # Crear una lista de números pares del 1 al 10
print(pares)

# Comprensión de listas con operaciones
cuadrados_pares = [x**2 for x in range(1, 11) if x % 2 == 0]  # Crear una lista con los cuadrados de los números pares del 1 al 10
print(cuadrados_pares)

# Desafío: Crear y manipular listas

# Ejercicio 1: Crear una lista con los números del 1 al 20 y luego crear una nueva lista que contenga solo los múltiplos de 3
numeros = list(range(1, 21))
multiplos_de_tres = [x for x in numeros if x % 3 == 0]
print(multiplos_de_tres)

# Ejercicio 2: Crear una lista con las primeras letras de cada palabra en una lista de palabras
palabras = ["Python", "es", "genial"]
primeras_letras = [palabra[0] for palabra in palabras]
print(primeras_letras)

# Proyecto: Implementar una función que almacene datos en una lista y los procese de alguna manera

class GestorDeNumeros:
    def __init__(self):
        self.lista_numeros = []

    def almacenar_datos(self, numeros):
        """Almacena una lista de números en la lista interna."""
        self.lista_numeros.extend(numeros)

    def obtener_numeros_pares(self):
        """Devuelve una lista de números pares almacenados."""
        return [num for num in self.lista_numeros if num % 2 == 0]

    def obtener_numeros_impares(self):
        """Devuelve una lista de números impares almacenados."""
        return [num for num in self.lista_numeros if num % 2 != 0]

    def obtener_suma_total(self):
        """Devuelve la suma de todos los números almacenados."""
        return sum(self.lista_numeros)

# Ejemplo de uso:
gestor = GestorDeNumeros()

# Almacenar una lista de números
gestor.almacenar_datos([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Obtener y mostrar números pares
numeros_pares = gestor.obtener_numeros_pares()
print("Números pares:", numeros_pares)

# Obtener y mostrar números impares
numeros_impares = gestor.obtener_numeros_impares()
print("Números impares:", numeros_impares)

# Obtener y mostrar la suma total de los números
suma_total = gestor.obtener_suma_total()
print("Suma total de los números:", suma_total)
