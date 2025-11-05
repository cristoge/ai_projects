# completa el problema de las 8 reinas.
#
# Utilitzant programació genetica, crea un algorisme que trobi una possible solució del problema de les 8 reinas
#
# (https://en.wikipedia.org/wiki/Eight_queens_puzzle) generalitzada per taules d’escacs de NxN.
#
# Hi ha moltes maneres d’aproximar aquest problema. intenta establir el Gen de l'individu solució de manera adient.
#
# I utilitza un fitness score de manera que la millor solució tingui un valor 0.
#

import random

"""
Implementar y definir el objeto individuo
"""


def generar_individuo(medida: int) -> list:
    """
    Crea un individuo que es representado en una lista de ints,
    el indice es la fila y el valor de la columna es donde se coloca la reina
    """

    nuevo_individuo = list(range(medida))
    random.shuffle(nuevo_individuo)
    return nuevo_individuo


"""
Definir la poblacion, un conjunto de individuos
"""


def generar_poblacion(medida_tablero: int, medida_poblacion: int) -> list:
    """
    Genera la poblacion inicial mediante comprension de listas.
    Args:
    medida_tablero: tamaño del medida_tablero
    medida_poblacion: numero de individuos
    Returns:
    Lista de individuos en este caso la poblacion inicial
    """
    poblacion = [generar_individuo(medida_tablero) for _ in range(medida_poblacion)]
    return poblacion


"""
Evalucion(fitness)
"""


def fitness(individuo: list):
    """
    Calcula el total de colisiones diagonales entre reinas
    Args:
        individuo: lista que representa una solucion
    Returns:
        Numero total de colisiones
    """
    colision = 0
    for i in range(len(individuo)):
        for j in range(i + 1, len(individuo)):
            if abs(i - j) == abs(individuo[i] - individuo[j]):
                colision += 1
    return colision


"""
Definir la seleccion, la reprodccion y la mutacion
"""


def evolucion(poblacion: list):
    """
    Ordena la población segun su numero de colisiones.

    Args:
        poblacion: lista de individuos.

    Returns:
        Lista de individuos ordenada por el numero de colisiones.
    """
    fitness_poblacion = [(i, fitness(i)) for i in poblacion]
    fitness_poblacion.sort(key=lambda x: x[1])
    return [x for x, _ in fitness_poblacion]


def crossover(padre1: list, padre2: list, mutation_chance=0.1):
    """
    Mezcla a los padres para generos 2 nuevos hijos, combinando la mitad
    de cada padre, haciendo que no haya columnas duplicadas.
    Args:
        padre1, padre2: listas que representan a los padres
        mutation_chance: probabilidad de mutacion de un hijo
    Returns:
        Una tupla con el resultado de 2 hijos
    """
    punto_medio = len(padre1) // 2
    hijo1 = padre1[:punto_medio]
    hijo2 = padre2[:punto_medio]
    for i in padre2:
        if i not in hijo1:
            hijo1.append(i)
    for i in padre1:
        if i not in hijo2:
            hijo2.append(i)

    return mutacion(hijo1, mutation_chance), mutacion(hijo2, mutation_chance)


def reproducir_todos_padres(poblacion: list):
    """
    Reproduce la poblacion, mezclando a los individuos por pares
    Args:
        poblacion: lista de padres seleccionados.
    Returns:
        Lista de nuevos hijos generados mediante crossover.
    """
    random.shuffle(poblacion)
    hijos = []
    for i in range(0, len(poblacion) - 1, 2):
        # -1 para que no se me vaya de posicion
        hijo1, hijo2 = crossover(poblacion[i], poblacion[i + 1])
        hijos.append(hijo1)
        hijos.append(hijo2)
    return hijos


def mutacion(individuo: list, mutation_chance) -> list:
    """
    Aplica una mutación random al individuo haciendo que se intercambien
    dos posiciones
    Args:
        individuo: lista de ints
        mutation_chance: probabilidad de mutación.
    Returns:
        Individuo posiblemente mutado.
    """
    if random.random() < mutation_chance:
        pos1 = random.randint(0, len(individuo) - 1)
        pos2 = random.randint(0, len(individuo) - 1)
        individuo[pos1], individuo[pos2] = individuo[pos2], individuo[pos1]
    return individuo


"""
Hilo principal
"""


def main():
    """
    Ejecuta el programa para el problema de las 8 reinas.
    -Definimos la medida de la poblacion y el tablero.
    -Aplica una seed para obtener los mismos resultados aleatorios en cada ejecucion
    -Genera una poblacion inicial.
    -Evalua y selecciona los mejores individuos.
    -Aplica reproduccion y mutacion.
    -Al encontrar la solucion con el fitness 0 o al alcanzar.
    el numero limite de generaciones finaliza el programa.
    """
    medida_poblacion = 10
    medida_tablero = 8
    random.seed(12)
    poblacion = generar_poblacion(medida_tablero, medida_poblacion)
    generaciones_maximas = 100000
    for i in range(generaciones_maximas):
        poblacion = evolucion(poblacion)
        mejor = poblacion[0]
        mejor_fitness = fitness(mejor)
        if mejor_fitness == 0:
            print(f"solucion encontrada en la generacion {i} y {mejor}")
            break
        padres = poblacion[: medida_poblacion // 2]
        hijos = reproducir_todos_padres(padres)
        poblacion = padres + hijos
        print("esta es la generacion ", i)


main()
