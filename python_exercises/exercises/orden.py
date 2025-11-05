co_data = ["first", 12, 47, 0, -4, 23.02, [2, 3, 4, 8, -3], "last", 3, 100, -31]


def clean_order(lista: list):
    clear_list = []
    for x in lista:
        if type(x) is int:
            clear_list.append(x)
    clear_list.sort()
    return clear_list


solucion = clean_order(co_data)
print(solucion)
