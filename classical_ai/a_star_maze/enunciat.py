import pyamaze as maze
from queue import PriorityQueue
import time

ROWS = 20
COLS = 20


def distance(cell1, cell2):
    """Calcula la distancia Manhattan entre dos celdas."""
    x1, y1 = cell1
    x2, y2 = cell2
    return abs(x1 - x2) + abs(y1 - y2)


def aStar(m):
    start = (m.rows, m.cols)
    goal = (1, 1)

    open_list = PriorityQueue()
    open_list.put((0, start))
    g_score = {cell: float("inf") for cell in m.grid}
    g_score[start] = 0
    f_score = {cell: float("inf") for cell in m.grid}
    f_score[start] = distance(start, goal)

    aPath = {}
    while not open_list.empty():
        current = open_list.get()[1]

        if current == goal:
            break

        for d in "ESNW":
            if m.maze_map[current][d] == True:
                if d == "E":
                    child = (current[0], current[1] + 1)
                elif d == "W":
                    child = (current[0], current[1] - 1)
                elif d == "N":
                    child = (current[0] - 1, current[1])
                elif d == "S":
                    child = (current[0] + 1, current[1])

                temp_g = g_score[current] + 1
                temp_f = temp_g + distance(child, goal)

                if temp_f < f_score[child]:
                    g_score[child] = temp_g
                    f_score[child] = temp_f
                    open_list.put((temp_f, child))
                    aPath[child] = current

    # reconstruir el camino
    fwdPath = {}
    cell = goal
    while cell != start:
        fwdPath[aPath[cell]] = cell
        cell = aPath[cell]
    return fwdPath


m = maze.maze(ROWS, COLS)
m.CreateMaze()

pre_Astar = time.time()
path = aStar(m)
post_Astar = time.time()
print("Tiempo de ejecución:", post_Astar - pre_Astar, "segundos")

a = maze.agent(m, footprints=True)
m.tracePath({a: path}, delay=50)
m.run()
