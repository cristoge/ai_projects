# Define una función "bus_stops" que:
#
# *   Acepte una lista de diccionarios con una serie de coordenadas ("x" e "y") con números enteros
# *   Verifique que todo son números enteros (en caso de no ser así que elimine las coordenadas que no lo cumplan)
# *   Calcule el orden que optimice la distancia mínima a recorrer si tenemos que pasar por todos los puntos.
# *   Muestre por pantalla la distancia mínima obtenida y devuelva un diccionario con el orden correcto de los puntos.
#
# Llama a la función con el diccionario "bs_data" e imprime su resultado por pantalla:
import math

bs_data = [
    {"x": 1, "y": 1},
    {"x": "some", "y": 12},
    {"x": 3, "y": 9},
    {"x": 9, "y": 4},
    {"x": 1, "y": 1},
    {"x": 1, "y": 5},
    {"x": 5, "y": 2},
    {"x": 4, "y": 10},
    {"x": 8, "y": 8},
    {"x": -3, "y": 2.3},
]


def only_ints(data: list):
    verify_data = list(
        filter(
            lambda coords: isinstance(coords["x"], int)
            and isinstance(coords["y"], int),
            data,
        )
    )
    return verify_data


def calculate_distance(coords1: dict, coords2: dict):
    distance = math.dist([coords1["x"], coords1["y"]], [coords2["x"], coords2["y"]])
    return distance


def bus_stops(bs_data: list):
    data = only_ints(bs_data)
    route = []
    total_distance = 0
    current_point = data[0]
    remaining_points = data[1:]
    while remaining_points:
        min_dist = math.inf
        near_point = None
        for i in remaining_points:
            dist = calculate_distance(current_point, i)
            if dist < min_dist:
                min_dist = dist
                near_point = i
        route.append(near_point)
        current_point = near_point
        total_distance += min_dist
        remaining_points.remove(near_point)
    print(f"route:{route}y la distancia es {total_distance}")


bus_stops(bs_data)
