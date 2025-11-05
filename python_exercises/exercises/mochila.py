# Define una función "backpack" que:

# *   Acepte dos parámetros de entrada: un número entero que especificará el peso que es capaz de soportar la mochila y una lista de pares de elementos (definidos en diccionarios) que contenga el peso y el valor de cada objeto
# *   Verifique que todos los parámetros son números enteros (los que no sean así serán eliminados)
# *   Calcule la mejor combinación de objetos que permita la máxima ganancia limitandose al peso que soporta la mochila
# *   Muestre por pantalla el valor obtenido
# *   Devuelva una lista con los elementos escogidos

# Llama a la función con los parámetros "backpack_weight_limit" y "objects" e imprime su resultado por pantalla:
#
# Correccion

backpack_weight_limit = 15
objects = [
    {"weight": 1, "value": 1},
    {"weight": 6, "value": 4},
    {"weight": 4, "value": 7},
    {"weight": 5, "value": 6},
    {"weight": 1, "value": 3},
    {"weight": 6, "value": 8},
    {"weight": 3, "value": 6},
    {"weight": 10, "value": 11},
    {"weight": 4, "value": 4},
    {"weight": 7, "value": 3},
]


# Función utilizada para buscar la mejor combinación comprobando todas las alternativas de manera recursiva


def calc_best_combination(weight_limit: int, objects: list) -> list:
    if weight_limit <= 0 or len(objects) == 0:
        return []
    primer_valor = objects[0]
    valores_restantes = objects[1:]
    if primer_valor["weight"] <= weight_limit:
        primero_incluido = [primer_valor] + calc_best_combination(
            weight_limit - primer_valor["weight"], valores_restantes
        )
    else:
        primero_incluido = []
    primero_excluido = calc_best_combination(weight_limit, valores_restantes)
    suma_incluido = sum(obj["value"] for obj in primero_incluido)
    suma_excluido = sum(obj["value"] for obj in primero_excluido)
    return primero_incluido if suma_incluido > suma_excluido else primero_excluido


lista_final = calc_best_combination(backpack_weight_limit, objects)
print(lista_final)
