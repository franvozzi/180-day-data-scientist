# Este es un ejemplo básico para mostrar el uso de sets en Python

# Crear un set en Python
s = set([5, 4, 6, 8, 8, 1])
print(s)       #{1, 4, 5, 6, 8}
print(type(s)) #<class 'set'>

# También se puede crear un set usando llaves {}
s = {5, 4, 6, 8, 8, 1}
print(s)       #{1, 4, 5, 6, 8}
print(type(s)) #<class 'set'>

# Operaciones con sets
s = set([5, 6, 7, 8])
# Intentar modificar un elemento da un TypeError
# s[0] = 3  # Error! TypeError

# Los sets no pueden contener elementos mutables como listas
# lista = ["Perú", "Argentina"]
# s = set(["México", "España", lista])  # Error! TypeError

# Iterar sobre un set
for ss in s:
    print(ss)  # 8, 5, 6, 7

# Con la función len() obtenemos la longitud del set
s = set([1, 2, 2, 3, 4])
print(len(s))  # 4

# Comprobar si un elemento está en el set
s = set(["Guitarra", "Bajo"])
print("Guitarra" in s)  # True

# Operaciones de conjuntos
s1 = set([1, 2, 3])
s2 = set([3, 4, 5])

# Unión de sets
print(s1 | s2)  # {1, 2, 3, 4, 5}
# Método union
print(s1.union(s2))  # {1, 2, 3, 4, 5}

# Métodos de sets
l = set([1, 2])
l.add(3)
print(l)  # {1, 2, 3}

s = set([1, 2])
s.remove(2)
print(s)  # {1}

s = set([1, 2])
s.discard(3)
print(s)  # {1, 2}

s = set([1, 2])
s.pop()
print(s)  # {2}

s = set([1, 2])
s.clear()
print(s)  # set()

# Más operaciones con sets
print(s1.intersection(s2))  # {3}

# Aquí podrían incluirse más ejemplos y métodos de sets según sea necesario

