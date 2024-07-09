# Definición del diccionario para almacenar datos del proyecto
proyecto = {
    'nombre': 'Sistema de Gestión de Inventarios',
    'descripcion': 'Una aplicación web para gestionar el inventario de productos.',
    'equipo': ['Ana', 'Carlos', 'Elena'],
    'version': 1.0
}

# Función para obtener la descripción del proyecto
def obtener_descripcion(proyecto):
    return proyecto.get('descripcion', 'Descripción no disponible')

# Función para actualizar el equipo del proyecto
def actualizar_equipo(proyecto, nuevo_equipo):
    proyecto['equipo'] = nuevo_equipo
    return proyecto

# Ejemplo de uso de las funciones
if __name__ == "__main__":
    print(obtener_descripcion(proyecto))  # Imprime la descripción del proyecto
    print(proyecto)  # Imprime el diccionario completo antes de actualizar el equipo
    proyecto = actualizar_equipo(proyecto, ['Ana', 'Carlos', 'Elena', 'Gabriel'])
    print(proyecto)  # Imprime el diccionario actualizado con el nuevo equipo
